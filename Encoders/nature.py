import torch.nn.functional as F
import torch.nn as nn
from mlagents.trainers.torch.layers import linear_layer, Initialization, Swish
from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch.model_serialization import exporting_to_onnx
from mlagents.trainers.torch.encoders import conv_output_shape

# ========================= MOE COMPONENTS FOR SIMPLE VISUAL ENCODER =========================

class AttentionExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, variant_id: int = 0):
        super().__init__()
        # Using a simpler but effective variant
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        enhanced_features = self.feature_transform(x)
        gate_values = self.gate(enhanced_features)
        attended_features = enhanced_features * gate_values
        return self.output_proj(attended_features)


class QFormerExpert(nn.Module):
    """Manual QFormer with learnable queries - ONNX opset 9 compatible"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, variant_id: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = 4  # 4 learnable queries
        
        # Learnable query parameters - 4 queries
        self.queries = nn.Parameter(torch.randn(self.num_queries, hidden_dim) * 0.02)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            Swish()
        )
        
        # Query-Key-Value projections (manual attention style)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)  
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output aggregation
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * self.num_queries, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Project input to hidden dimension
        input_features = self.input_proj(x)  # [B, hidden_dim]
        input_features = input_features.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Expand queries for batch
        queries = self.queries.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, num_queries, hidden_dim]
        
        # Compute Q, K, V (queries attend to input)
        Q = self.q_proj(queries)  # [B, num_queries, hidden_dim]
        K = self.k_proj(input_features)  # [B, 1, hidden_dim] 
        V = self.v_proj(input_features)  # [B, 1, hidden_dim]
        
        # Manual attention computation
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # [B, num_queries, 1]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_queries, 1]
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [B, num_queries, hidden_dim]
        
        # Add residual connection and normalize
        queries_updated = self.norm(queries + attended)  # [B, num_queries, hidden_dim]
        
        # Flatten and project to output
        flattened = queries_updated.reshape(batch_size, self.num_queries * self.hidden_dim)  # [B, num_queries * hidden_dim]
        output = self.output_proj(flattened)  # [B, output_dim]
        
        return output


class MLPExpert(nn.Module):
    """Simple MLP Expert - ONNX opset 9 compatible"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, variant_id: int = 0):
        super().__init__()
        
        # Standard MLP with residual connection
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Residual projection if dimensions don't match
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        mlp_output = self.mlp(x)
        residual = self.residual_proj(x)
        return mlp_output + residual  # Residual connection


class SimpleDiverseGate(nn.Module):
    """Simple gating network for 2 experts"""
    def __init__(self, input_dim: int, num_experts: int = 2):
        super().__init__()
        self.num_experts = num_experts
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            Swish(),  # Better than ReLU, ONNX compatible
            nn.Linear(input_dim // 2, num_experts)
        )
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor):
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits / torch.clamp(self.temperature, min=0.1), dim=-1)
        
        # Load balancing loss  
        expert_importance = torch.mean(gate_weights, dim=0)
        load_balance_loss = torch.var(expert_importance) * self.num_experts
        
        return gate_weights, load_balance_loss


class AttentionQFormerMoELayer(nn.Module):
    """MoE layer with 2 experts: MLP + Attention + residual connection (lightweight)"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
            
        # Create 2 experts: MLP + Attention (lightweight)  
        self.experts = nn.ModuleList([
            MLPExpert(input_dim, hidden_dim, output_dim),
            AttentionExpert(input_dim, hidden_dim, output_dim)
        ])
        self.num_experts = 2
        
        # Gating network
        self.gate = SimpleDiverseGate(input_dim, self.num_experts)
        
        # Residual projection if dimensions don't match
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor):
        # Get gating weights
        gate_weights, load_balance_loss = self.gate(x)
        
        # Apply all experts
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)
            expert_outputs.append(expert_out)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, output_dim]
        
        # Weighted combination
        gate_weights = gate_weights.unsqueeze(-1)  # [B, num_experts, 1]
        output = torch.sum(gate_weights * expert_outputs, dim=1)  # [B, output_dim]
        
        # Residual connection
        residual = self.residual_proj(x)
        output = output + residual
        
        return output, load_balance_loss


class NatureVisualEncoder(nn.Module):
    """
    Enhanced SimpleVisualEncoder with MoE layers
    Features:
    - Original CNN backbone (conv_layers + dense) - loads from existing checkpoints
    - 3 additional MoE layers with residual connections
    - MoE loss tracking for training
    """
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        self.h_size = output_size
        conv_1_hw = conv_output_shape((height, width), 8, 4)
        conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32

        # ===== ORIGINAL LAYERS (will load from checkpoint) =====
        self.conv_layers = nn.Sequential(
            nn.Conv2d(initial_channels, 16, [8, 8], [4, 4]),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, [4, 4], [2, 2]),
            nn.LeakyReLU(),
        )
        self.dense = nn.Sequential(
            linear_layer(
                self.final_flat,
                self.h_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,  # Use ReLU gain
            ),
            nn.LeakyReLU(),
        )
        
        # ===== NEW MoE LAYERS (will be randomly initialized) =====
        # 3 additional MoE layers with residual connections
        self.moe_layer1 = AttentionQFormerMoELayer(output_size, output_size, output_size)
        self.moe_layer2 = AttentionQFormerMoELayer(output_size, output_size, output_size) 
        self.moe_layer3 = AttentionQFormerMoELayer(output_size, output_size, output_size)
        
        # Layer norms for stability
        self.norm1 = nn.LayerNorm(output_size)
        self.norm2 = nn.LayerNorm(output_size)
        self.norm3 = nn.LayerNorm(output_size)
        
        # MoE loss tracking
        self.moe_loss = 0.0

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])
        hidden = self.conv_layers(visual_obs)
        hidden = hidden.reshape(-1, self.final_flat)
        
        # Original dense layer
        x = self.dense(hidden)  # [B, output_size]
        
        # Apply 3 MoE layers with residual connections
        x1, loss1 = self.moe_layer1(x)
        x1 = self.norm1(x1)
        
        x2, loss2 = self.moe_layer2(x1) 
        x2 = self.norm2(x2)
        
        x3, loss3 = self.moe_layer3(x2)
        x3 = self.norm3(x3)
        
        # Track MoE loss
        if self.training:
            self.moe_loss = loss1 + loss2 + loss3
        
        return x3
    
    def get_moe_loss(self):
        """Get and reset MoE loss"""
        loss = self.moe_loss
        self.moe_loss = 0.0
        return loss

# class NatureVisualEncoder(nn.Module):
#     """
#     Enhanced SimpleVisualEncoder with MoE layers
#     Features:
#     - Original CNN backbone (conv_layers + dense) - loads from existing checkpoints
#     - 2 additional MoE layers with residual connections
#     - MoE loss tracking for training
#     """
#     def __init__(
#         self, height: int, width: int, initial_channels: int, output_size: int
#     ):
#         super().__init__()
#         self.h_size = output_size
#         conv_1_hw = conv_output_shape((height, width), 8, 4)
#         conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
#         self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32

#         # ===== ORIGINAL LAYERS (will load from checkpoint) =====
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(initial_channels, 16, [8, 8], [4, 4]),
#             nn.LeakyReLU(),
#             nn.Conv2d(16, 32, [4, 4], [2, 2]),
#             nn.LeakyReLU(),
#         )
#         self.dense = nn.Sequential(
#             linear_layer(
#                 self.final_flat,
#                 self.h_size,
#                 kernel_init=Initialization.KaimingHeNormal,
#                 kernel_gain=1.41,  # Use ReLU gain
#             ),
#             nn.LeakyReLU(),
#         )

#         # ===== NEW MoE LAYERS (2 layers) =====
#         self.moe_layer1 = AttentionQFormerMoELayer(output_size, output_size, output_size)
#         self.moe_layer2 = AttentionQFormerMoELayer(output_size, output_size, output_size)

#         # Layer norms for stability
#         self.norm1 = nn.LayerNorm(output_size)
#         self.norm2 = nn.LayerNorm(output_size)

#         # MoE loss tracking
#         self.moe_loss = 0.0

#     def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
#         if not exporting_to_onnx.is_exporting():
#             visual_obs = visual_obs.permute([0, 3, 1, 2])
#         hidden = self.conv_layers(visual_obs)
#         hidden = hidden.reshape(-1, self.final_flat)

#         # Original dense layer
#         x = self.dense(hidden)  # [B, output_size]

#         # ===== 2 MoE layers =====
#         x1, loss1 = self.moe_layer1(x)
#         x1 = self.norm1(x1)

#         x2, loss2 = self.moe_layer2(x1)
#         x2 = self.norm2(x2)

#         # Track MoE loss
#         if self.training:
#             self.moe_loss = loss1 + loss2

#         return x2

#     def get_moe_loss(self):
#         """Get and reset MoE loss"""
#         loss = self.moe_loss
#         self.moe_loss = 0.0
#         return loss
