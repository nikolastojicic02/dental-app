import torch
import torch.nn as nn
import torch.nn.functional as F

# Detection model architecture (from your code)
class PointNetSetAbstraction(nn.Module):
    """Set Abstraction module from PointNet++"""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        B, N, C_xyz = xyz.shape

        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)
            xyz_trans = xyz.transpose(1, 2)

            if points is not None:
                new_points = torch.cat([xyz_trans, points], dim=1)
            else:
                new_points = xyz_trans

            new_points = new_points.unsqueeze(2)
        else:
            if self.npoint >= N:
                sample_idx = torch.arange(N).unsqueeze(0).repeat(B, 1).to(xyz.device)
                new_xyz = xyz
            else:
                sample_idx = torch.randint(0, N, (B, self.npoint), device=xyz.device)
                new_xyz = torch.gather(xyz, 1, sample_idx.unsqueeze(-1).expand(-1, -1, 3))

            actual_npoint = new_xyz.shape[1]
            n_sample = min(self.nsample, N)

            neighbor_idx = torch.randint(0, N, (B, actual_npoint, n_sample), device=xyz.device)

            neighbor_xyz = torch.gather(
                xyz.unsqueeze(1).expand(-1, actual_npoint, -1, -1),
                2,
                neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            )

            relative_xyz = neighbor_xyz - new_xyz.unsqueeze(2)
            relative_xyz = relative_xyz.permute(0, 3, 1, 2)

            if points is not None:
                neighbor_features = torch.gather(
                    points.unsqueeze(2).expand(-1, -1, actual_npoint, -1),
                    3,
                    neighbor_idx.unsqueeze(1).expand(-1, points.shape[1], -1, -1)
                )
                new_points = torch.cat([relative_xyz, neighbor_features], dim=1)
            else:
                new_points = relative_xyz

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, dim=-1)[0]

        return new_xyz, new_points


class BBoxDetectionHead(nn.Module):
    """Dual-head output for bounding box prediction"""

    def __init__(self, feature_dim):
        super().__init__()

        self.center_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

        self.rotation_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, features):
        center = self.center_head(features)
        rotation_6d = self.rotation_head(features)
        return center, rotation_6d


class PointNetPlusBBoxDetector(nn.Module):
    """PointNet++ for 3D bounding box detection"""

    def __init__(self, feature_dim=256):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=2.0, nsample=64,
            in_channel=6, mlp=[64, 64, 128]
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=4.0, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256]
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, feature_dim],
            group_all=True
        )

        self.bbox_head = BBoxDetectionHead(feature_dim)

    def forward(self, data):
        xyz = data[:, :, :3]
        normals = data[:, :, 3:].transpose(1, 2)

        l1_xyz, l1_points = self.sa1(xyz, normals)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        global_features = l3_points.squeeze(-1) if l3_points.dim() == 3 else l3_points
        if global_features.dim() > 2:
            global_features = global_features.view(global_features.size(0), -1)

        center, rotation_6d = self.bbox_head(global_features)

        return center, rotation_6d


# Segmentation model architecture
class PointNetFeaturePropagation(nn.Module):
    """Feature Propagation layer - upsampling"""

    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        B, N1, _ = xyz1.shape
        _, N2, _ = xyz2.shape

        if N2 == 1:
            interpolated_points = points2.expand(-1, -1, N1)
        else:
            dists = torch.cdist(xyz1, xyz2)
            knn_idx = torch.topk(dists, k=min(3, N2), largest=False, dim=2)[1]
            knn_dists = torch.gather(dists, 2, knn_idx)
            knn_dists = torch.clamp(knn_dists, min=1e-10)

            weights = 1.0 / knn_dists
            weights = weights / weights.sum(dim=2, keepdim=True)

            knn_feats = torch.gather(
                points2.unsqueeze(2).expand(-1, -1, N1, -1),
                3,
                knn_idx.unsqueeze(1).expand(-1, points2.shape[1], -1, -1)
            )

            interpolated_points = (knn_feats * weights.unsqueeze(1)).sum(dim=3)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points


class PointNetPlusPlusSegmentation(nn.Module):
    """PointNet++ for Part Segmentation"""

    def __init__(self, num_classes=7, feature_dim=256):
        super().__init__()

        # Encoder
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=2.0, nsample=32,
            in_channel=6, mlp=[64, 64, 128]
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=4.0, nsample=32,
            in_channel=128 + 3, mlp=[128, 128, 256]
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, feature_dim],
            group_all=True
        )

        # Decoder
        self.fp3 = PointNetFeaturePropagation(
            in_channel=feature_dim + 256, mlp=[256, 256]
        )

        self.fp2 = PointNetFeaturePropagation(
            in_channel=256 + 128, mlp=[256, 128]
        )

        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + 6, mlp=[128, 128, 128]
        )

        # Classification head
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, features):
        xyz = features[:, :, :3]
        points = features[:, :, 3:].transpose(1, 2)

        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, features.transpose(1, 2), l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(x)
        logits = self.conv2(x)

        return logits
