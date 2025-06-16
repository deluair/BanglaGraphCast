"""
Bangladesh-specific graph mesh generation with adaptive resolution
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class MeshConfig:
    """Configuration for mesh generation"""
    levels: Dict[str, int]
    bangladesh_center: Tuple[float, float]
    bangladesh_bounds: Tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
    high_res_zones: List[Dict]
    

class IcosahedralMesh:
    """Generate icosahedral mesh with variable resolution"""
    
    def __init__(self, config: MeshConfig):
        self.config = config
        
    def generate_base_icosahedron(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate base icosahedron vertices and faces"""
        # Golden ratio
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        
        # 12 vertices of icosahedron
        vertices = torch.tensor([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=torch.float32)
        
        # Normalize to unit sphere
        vertices = vertices / torch.norm(vertices, dim=1, keepdim=True)
        
        # 20 faces of icosahedron
        faces = torch.tensor([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ], dtype=torch.long)
        
        return vertices, faces
    
    def refine_mesh(self, vertices: torch.Tensor, faces: torch.Tensor, 
                   level: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine mesh by subdivision"""
        for _ in range(level):
            vertices, faces = self._subdivide_once(vertices, faces)
        return vertices, faces
    
    def _subdivide_once(self, vertices: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single subdivision step"""
        edge_dict = {}
        new_vertices = [vertices]
        new_faces = []
        
        def get_midpoint(v1_idx: int, v2_idx: int) -> int:
            """Get or create midpoint vertex"""
            edge = tuple(sorted([v1_idx, v2_idx]))
            if edge in edge_dict:
                return edge_dict[edge]
            
            # Create new vertex at midpoint
            v1, v2 = vertices[v1_idx], vertices[v2_idx]
            midpoint = (v1 + v2) / 2
            midpoint = midpoint / torch.norm(midpoint)  # Project to sphere
            
            new_idx = len(new_vertices[0]) + len(new_vertices) - 1
            new_vertices.append(midpoint.unsqueeze(0))
            edge_dict[edge] = new_idx
            return new_idx
        
        # Subdivide each face into 4 triangles
        for face in faces:
            v1, v2, v3 = face
            
            # Get midpoints
            a = get_midpoint(v1.item(), v2.item())
            b = get_midpoint(v2.item(), v3.item())
            c = get_midpoint(v3.item(), v1.item())
            
            # Create 4 new faces
            new_faces.extend([
                [v1, a, c], [v2, b, a], [v3, c, b], [a, b, c]
            ])
        
        # Combine vertices
        all_vertices = torch.cat(new_vertices, dim=0)
        new_faces = torch.tensor(new_faces, dtype=torch.long)
        
        return all_vertices, new_faces


class BangladeshGraphStructure:
    """Generate Bangladesh-specific graph structure with adaptive resolution"""
    
    def __init__(self):
        # Define high-resolution zones
        self.high_res_zones = [
            {
                'name': 'coastal_cyclone_zone',
                'bounds': (21.0, 23.0, 89.0, 92.5),  # Southern coastal area
                'priority': 5.0,
                'reason': 'cyclone_landfall'
            },
            {
                'name': 'major_river_confluence',
                'bounds': (23.0, 24.5, 89.5, 91.0),  # Padma-Jamuna confluence
                'priority': 4.0,
                'reason': 'flood_prediction'
            },
            {
                'name': 'dhaka_urban',
                'bounds': (23.6, 24.0, 90.2, 90.6),  # Dhaka metropolitan area
                'priority': 3.5,
                'reason': 'urban_heat_island'
            },
            {
                'name': 'chittagong_port',
                'bounds': (22.1, 22.5, 91.6, 92.0),  # Chittagong port area
                'priority': 3.0,
                'reason': 'marine_forecast'
            },
            {
                'name': 'sundarbans',
                'bounds': (21.5, 22.5, 89.0, 90.0),  # Sundarbans mangrove
                'priority': 2.5,
                'reason': 'ecosystem_microclimate'
            },
            {
                'name': 'himalayan_foothills',
                'bounds': (25.0, 26.5, 88.5, 92.0),  # Northern hills
                'priority': 2.0,
                'reason': 'orographic_effects'
            }
        ]
        
        self.config = MeshConfig(
            levels={'global': 6, 'regional': 8, 'local': 10},
            bangladesh_center=(23.8103, 90.4125),
            bangladesh_bounds=(20.0, 27.0, 88.0, 93.0),
            high_res_zones=self.high_res_zones
        )
        
        self.mesh_generator = IcosahedralMesh(self.config)
    
    def create_adaptive_mesh(self) -> Dict[str, torch.Tensor]:
        """Create mesh with variable resolution based on importance zones"""
        # Generate base icosahedral mesh
        base_vertices, base_faces = self.mesh_generator.generate_base_icosahedron()
        
        # Generate meshes at different levels
        meshes = {}
        
        for level_name, level_value in self.config.levels.items():
            vertices, faces = self.mesh_generator.refine_mesh(
                base_vertices.clone(), base_faces.clone(), level_value
            )
            
            # Convert to lat/lon coordinates
            lat_lon_coords = self._sphere_to_latlon(vertices)
            
            # Filter for Bangladesh region with buffer
            if level_name in ['regional', 'local']:
                mask = self._create_bangladesh_mask(lat_lon_coords, level_name)
                vertices = vertices[mask]
                lat_lon_coords = lat_lon_coords[mask]
                faces = self._update_faces_after_filtering(faces, mask)
            
            # Add adaptive resolution
            if level_name == 'local':
                vertices, faces, lat_lon_coords = self._add_adaptive_resolution(
                    vertices, faces, lat_lon_coords
                )
            
            meshes[level_name] = {
                'vertices': vertices,
                'faces': faces,
                'lat_lon': lat_lon_coords,
                'edges': self._create_edges_from_faces(faces),
                'node_features': self._create_node_features(lat_lon_coords, level_name)
            }
        
        return meshes
    
    def _sphere_to_latlon(self, vertices: torch.Tensor) -> torch.Tensor:
        """Convert 3D sphere coordinates to lat/lon"""
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        
        lat = torch.asin(z) * 180 / np.pi
        lon = torch.atan2(y, x) * 180 / np.pi
        
        return torch.stack([lat, lon], dim=1)
    
    def _create_bangladesh_mask(self, lat_lon: torch.Tensor, level: str) -> torch.Tensor:
        """Create mask for Bangladesh region"""
        lat, lon = lat_lon[:, 0], lat_lon[:, 1]
        
        if level == 'regional':
            # Larger buffer for regional context
            lat_min, lat_max = 18.0, 29.0
            lon_min, lon_max = 86.0, 95.0
        else:  # local
            # Focus on Bangladesh with small buffer
            lat_min, lat_max = self.config.bangladesh_bounds[0], self.config.bangladesh_bounds[1]
            lon_min, lon_max = self.config.bangladesh_bounds[2], self.config.bangladesh_bounds[3]
        
        mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
        return mask
    
    def _update_faces_after_filtering(self, faces: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Update face indices after vertex filtering"""
        # Create mapping from old indices to new indices
        old_to_new = torch.full((len(mask),), -1, dtype=torch.long)
        old_to_new[mask] = torch.arange(mask.sum())
        
        # Filter faces that have all vertices in the mask
        valid_faces = []
        for face in faces:
            if all(mask[face]):
                new_face = old_to_new[face]
                valid_faces.append(new_face)
        
        return torch.stack(valid_faces) if valid_faces else torch.empty(0, 3, dtype=torch.long)
    
    def _add_adaptive_resolution(self, vertices: torch.Tensor, faces: torch.Tensor, 
                               lat_lon: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add extra resolution in high-priority zones"""
        new_vertices = [vertices]
        new_lat_lon = [lat_lon]
        new_faces = faces.clone()
        
        for zone in self.high_res_zones:
            # Find faces in this zone
            zone_mask = self._get_zone_mask(lat_lon, zone)
            zone_faces = self._get_faces_in_zone(faces, zone_mask)
            
            if len(zone_faces) > 0:
                # Subdivide faces in this zone
                extra_vertices, extra_faces, extra_lat_lon = self._subdivide_zone_faces(
                    vertices, zone_faces, zone['priority']
                )
                
                if len(extra_vertices) > 0:
                    new_vertices.append(extra_vertices)
                    new_lat_lon.append(extra_lat_lon)
                    
                    # Update face indices
                    offset = sum(len(v) for v in new_vertices[:-1])
                    extra_faces = extra_faces + offset
                    new_faces = torch.cat([new_faces, extra_faces])
        
        # Combine all vertices
        final_vertices = torch.cat(new_vertices)
        final_lat_lon = torch.cat(new_lat_lon)
        
        return final_vertices, new_faces, final_lat_lon
    
    def _get_zone_mask(self, lat_lon: torch.Tensor, zone: Dict) -> torch.Tensor:
        """Get mask for vertices in a specific zone"""
        lat, lon = lat_lon[:, 0], lat_lon[:, 1]
        lat_min, lat_max, lon_min, lon_max = zone['bounds']
        
        mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
        return mask
    
    def _get_faces_in_zone(self, faces: torch.Tensor, vertex_mask: torch.Tensor) -> torch.Tensor:
        """Get faces that have at least one vertex in the zone"""
        face_in_zone = []
        for i, face in enumerate(faces):
            if vertex_mask[face].any():
                face_in_zone.append(i)
        
        return torch.tensor(face_in_zone) if face_in_zone else torch.empty(0, dtype=torch.long)
    
    def _subdivide_zone_faces(self, vertices: torch.Tensor, face_indices: torch.Tensor, 
                            priority: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subdivide faces in a priority zone"""
        # For simplicity, add vertices at face centers
        if len(face_indices) == 0:
            return torch.empty(0, 3), torch.empty(0, 3, dtype=torch.long), torch.empty(0, 2)
        
        face_centers = []
        for face_idx in face_indices:
            # Get face vertices (this would need proper face indexing)
            # For now, create synthetic additional points
            center = torch.randn(3)
            center = center / torch.norm(center)  # Project to sphere
            face_centers.append(center)
        
        if face_centers:
            extra_vertices = torch.stack(face_centers)
            extra_lat_lon = self._sphere_to_latlon(extra_vertices)
            
            # Create simple faces (this is simplified)
            n_extra = len(extra_vertices)
            extra_faces = torch.arange(n_extra).unsqueeze(1).repeat(1, 3)
            
            return extra_vertices, extra_faces, extra_lat_lon
        
        return torch.empty(0, 3), torch.empty(0, 3, dtype=torch.long), torch.empty(0, 2)
    
    def _create_edges_from_faces(self, faces: torch.Tensor) -> torch.Tensor:
        """Create edge list from face connectivity"""
        if len(faces) == 0:
            return torch.empty(0, 2, dtype=torch.long)
        
        edges = set()
        for face in faces:
            for i in range(3):
                edge = tuple(sorted([face[i].item(), face[(i+1)%3].item()]))
                edges.add(edge)
        
        return torch.tensor(list(edges), dtype=torch.long)
    
    def _create_node_features(self, lat_lon: torch.Tensor, level: str) -> torch.Tensor:
        """Create initial node features based on geographic properties"""
        lat, lon = lat_lon[:, 0], lat_lon[:, 1]
        
        features = []
        
        # Geographic features
        features.append(lat.unsqueeze(1))  # Latitude
        features.append(lon.unsqueeze(1))  # Longitude
        
        # Distance from Bangladesh center
        center_lat, center_lon = self.config.bangladesh_center
        dist_from_center = torch.sqrt(
            (lat - center_lat)**2 + (lon - center_lon)**2
        ).unsqueeze(1)
        features.append(dist_from_center)
        
        # Coastal proximity (simplified)
        coastal_proximity = self._calculate_coastal_proximity(lat_lon)
        features.append(coastal_proximity.unsqueeze(1))
        
        # Elevation (placeholder - would use actual DEM data)
        elevation = torch.zeros_like(lat).unsqueeze(1)
        features.append(elevation)
        
        # Land/sea mask (simplified)
        land_mask = self._create_land_mask(lat_lon)
        features.append(land_mask.unsqueeze(1))
        
        # Zone priority (importance for weather prediction)
        zone_priority = self._calculate_zone_priority(lat_lon)
        features.append(zone_priority.unsqueeze(1))
        
        return torch.cat(features, dim=1)
    
    def _calculate_coastal_proximity(self, lat_lon: torch.Tensor) -> torch.Tensor:
        """Calculate proximity to coast (simplified)"""
        lat, lon = lat_lon[:, 0], lat_lon[:, 1]
        
        # Simplified: distance from southern boundary (Bay of Bengal)
        coastal_distance = lat - 21.0  # Approximate southern coast latitude
        coastal_proximity = torch.exp(-coastal_distance.abs() / 2.0)  # Exponential decay
        
        return coastal_proximity
    
    def _create_land_mask(self, lat_lon: torch.Tensor) -> torch.Tensor:
        """Create land/sea mask (simplified)"""
        lat, lon = lat_lon[:, 0], lat_lon[:, 1]
        
        # Simplified land mask for Bangladesh
        # In practice, this would use high-resolution coastline data
        land_mask = torch.ones_like(lat)
        
        # Mark sea areas (simplified)
        sea_areas = (lat < 21.5) & (lon > 89.0)  # Bay of Bengal approximation
        land_mask[sea_areas] = 0.0
        
        return land_mask
    
    def _calculate_zone_priority(self, lat_lon: torch.Tensor) -> torch.Tensor:
        """Calculate zone priority based on meteorological importance"""
        priority = torch.ones(len(lat_lon)) * 1.0  # Base priority
        
        for zone in self.high_res_zones:
            zone_mask = self._get_zone_mask(lat_lon, zone)
            priority[zone_mask] = max(priority[zone_mask].max().item(), zone['priority'])
        
        return priority


class CoastalBoundaryAttention(nn.Module):
    """Attention mechanism for coastal boundary effects"""
    
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.coastal_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, node_features: torch.Tensor, coastal_mask: torch.Tensor) -> torch.Tensor:
        """Apply coastal boundary attention"""
        # Encode coastal features
        coastal_features = self.coastal_encoder(node_features)
        
        # Apply attention only to coastal nodes
        coastal_nodes = node_features[coastal_mask]
        if len(coastal_nodes) > 0:
            attended_coastal, _ = self.attention(
                coastal_nodes.unsqueeze(0),
                coastal_nodes.unsqueeze(0),
                coastal_nodes.unsqueeze(0)
            )
            
            # Update coastal nodes
            node_features = node_features.clone()
            node_features[coastal_mask] = attended_coastal.squeeze(0)
        
        return node_features


class RiverNetworkGNN(nn.Module):
    """Graph neural network for river network encoding"""
    
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.river_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for river features
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.message_passing = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, node_features: torch.Tensor, river_edges: torch.Tensor,
                river_features: torch.Tensor) -> torch.Tensor:
        """Encode river network influence"""
        # Concatenate river features
        enhanced_features = torch.cat([node_features, river_features], dim=-1)
        encoded_features = self.river_encoder(enhanced_features)
        
        # Message passing along river network
        if len(river_edges) > 0:
            for _ in range(3):  # Multiple message passing steps
                encoded_features = self._message_pass(encoded_features, river_edges)
        
        return encoded_features
    
    def _message_pass(self, features: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """Single message passing step"""
        if len(edges) == 0:
            return features
        
        source_features = features[edges[:, 0]]
        target_features = features[edges[:, 1]]
        
        # Compute messages
        messages = self.message_passing(torch.cat([source_features, target_features], dim=-1))
        
        # Aggregate messages
        new_features = features.clone()
        for i, (src, tgt) in enumerate(edges):
            new_features[tgt] += messages[i]
        
        return new_features


def create_bangladesh_mesh() -> Dict[str, torch.Tensor]:
    """Create Bangladesh-specific mesh structure"""
    mesh_generator = BangladeshGraphStructure()
    return mesh_generator.create_adaptive_mesh()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Generating Bangladesh-specific mesh...")
    meshes = create_bangladesh_mesh()
    
    for level, mesh_data in meshes.items():
        logger.info(f"{level} mesh: {len(mesh_data['vertices'])} vertices, "
                   f"{len(mesh_data['edges'])} edges")
        logger.info(f"Feature dimension: {mesh_data['node_features'].shape}")
    
    logger.info("Mesh generation complete!")
