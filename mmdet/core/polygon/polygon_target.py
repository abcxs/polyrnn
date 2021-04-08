import torch
from torch.nn.modules.utils import _pair
import numpy as np
import shapely.geometry as shgeo
from mmdet.core import multi_apply
from .util import poly01_to_poly0g, xy_to_class, get_edge_mask
# 导入错误，猜测是由于循环导入，暂时copy
from .gaussian_target import gen_gaussian_target
import random

EPS = 1e-7


def polygon_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_polygons_list, cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    proposal_inds_list, polygon_target_list, polygon_mask_list, vertex_target_list, polygon_vertex_target_list, edge_target_list, offset_target_list = multi_apply(
        polygon_target_single, pos_proposals_list, pos_assigned_gt_inds_list, gt_polygons_list, cfg_list)
    polygon_targets = torch.cat(polygon_target_list, dim=0)
    polygon_masks = torch.cat(polygon_mask_list, dim=0)
    vertex_targets = torch.cat(vertex_target_list, dim=0)
    polygon_vertex_targets = torch.cat(polygon_vertex_target_list, dim=0)
    edge_targets = torch.cat(edge_target_list, dim=0)
    offset_targets = torch.cat(offset_target_list, dim=0)
    return proposal_inds_list, polygon_targets, polygon_masks, vertex_targets, polygon_vertex_targets, edge_targets, offset_targets


def polygon_target_single(pos_proposals, pos_assigned_gt_inds, gt_polygons, cfg):
    device = pos_proposals.device
    num_pos = pos_proposals.size(0)

    polygon_size = _pair(cfg.polygon_size)
    p_h, p_w = polygon_size
    # 是否过滤
    # 阈值与否
    # 可进行调参
    filter_multi_part = cfg.filter_multi_part
    poly_iou_thresh = cfg.poly_iou_thresh
    radius = cfg.poly_radius
    max_polyon_len = cfg.max_polygon_len
    epsilon = cfg.get('epsilon', 1)
    remove_abundant = cfg.get('remove_abundant', False)
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

        polygon_target = []
        proposal_inds = []
        polygon_mask = []
        polygon_vertex_target = []
        vertex_target = []
        edge_target = []
        offset_target = []
        for i, (ind, proposal) in enumerate(zip(pos_assigned_gt_inds, proposals_np)):
            poly_ins = shgeo.Polygon(gt_polygons[ind])
            x1, y1, x2, y2 = proposal[:4]
            poly_proposal = shgeo.Polygon(
                [[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
            
            if not (poly_ins.is_valid and poly_proposal.is_valid):
                continue
                
            inter_polys = poly_ins.intersection(poly_proposal)
    
            if type(inter_polys) != shgeo.polygon.Polygon and type(inter_polys) != shgeo.multipolygon.MultiPolygon:
                continue
            if filter_multi_part and type(inter_polys) == shgeo.multipolygon.MultiPolygon:
                continue
            if type(inter_polys) == shgeo.polygon.Polygon:
                inter_polys = [inter_polys]

            for inter_poly in inter_polys:
                if inter_poly.area / poly_ins.area < poly_iou_thresh:
                    continue
                # 由于是笛卡尔坐标系，而传入的是opencv坐标系的结果，设置sign为正数，从而获得顺时针
                try:
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                except Exception as e:
                    print(str(e))
                    print(type(inter_poly), type(inter_polys))
                    continue
                    
                temp_inter_poly = (np.array(list(inter_poly.exterior.coords)[
                                   :-1]) - [x1, y1]) / [x2 - x1 + EPS, y2 - y1 + EPS]
                # 投影到grid_size之后，不会出现满值
                # 例如：1映射之后为grid_size，会造成越界，故限定[0, grid_size)
                temp_inter_poly = np.clip(temp_inter_poly, 0, 1 - EPS)

                temp_inter_poly, offset = poly01_to_poly0g(
                    temp_inter_poly, polygon_size, epsilon=epsilon, remove_abundant=remove_abundant)
                if len(temp_inter_poly) < 3:
                    continue
                roll_index = random.choice(range(temp_inter_poly.shape[0]))
                temp_inter_poly = np.roll(temp_inter_poly, roll_index, axis=0)
                offset = np.roll(offset, roll_index, axis=0)

                poly_ = np.ones([max_polyon_len, 2]) * -1

                length = min(len(temp_inter_poly), max_polyon_len)
                poly_[:length] = temp_inter_poly[:length]
                polygon_target.append(poly_)
                
                offset_ = np.zeros([max_polyon_len, 2])
                offset_[:length] = offset[:length]
                offset_target.append(offset_)

                temp_mask = np.zeros(max_polyon_len)
                # 同时训练最后一个节点，即为终止点
                temp_mask[:length + 1] = 1
                polygon_mask.append(temp_mask)

                proposal_inds.append(i)
                
                edge_mask = np.zeros(polygon_size)
                get_edge_mask(temp_inter_poly, edge_mask)
                edge_target.append(edge_mask)
                
                vertex_mask = torch.zeros(polygon_size, dtype=torch.float32)
                polygon_vertex = torch.zeros(max_polyon_len, p_h * p_w + 1, dtype=torch.float32)
                for j, p in enumerate(temp_inter_poly):
                    if j < max_polyon_len:
                        poly_vertex = polygon_vertex[j][:p_h * p_w].reshape(p_h, p_w)
                        poly_vertex = gen_gaussian_target(poly_vertex, p, radius)
                        polygon_vertex[j][:p_h * p_w] = poly_vertex.flatten()
                    vertex_mask = gen_gaussian_target(vertex_mask, p, radius)
                polygon_vertex[length:, -1] = 1
                vertex_target.append(vertex_mask)
                polygon_vertex_target.append(polygon_vertex)
        
        polygon_vertex_target = torch.stack(polygon_vertex_target)
        polygon_target = np.array(
            polygon_target).reshape(-1, max_polyon_len, 2)
        proposal_inds = np.array(proposal_inds, dtype=np.long)
#         assert proposal_inds.min() >= 0 and proposal_inds.max() < len(proposals_np), f'{proposal_inds.min()}, {proposal_inds.max()}, {len(proposals_np)}'
        polygon_mask = np.array(polygon_mask).reshape(-1, max_polyon_len)
        edge_target = np.array(edge_target)
        offset_target = np.array(offset_target)
    else:
        polygon_vertex_target = torch.empty(0, max_polygon_len, p_h * p_w + 1)
        polygon_target = np.empty((0, max_polyon_len, 2))
        proposal_inds = np.array([], dtype=np.long)
        polygon_mask = np.empty((0, max_polyon_len))
        edge_target = np.empty((0, p_h, p_w))
        offset_target = np.empty((0, max_poly_len, 2))
    polygon_vertex_target = polygon_vertex_target.to(device)
    polygon_target = torch.from_numpy(polygon_target).float().to(device)
    polygon_mask = torch.from_numpy(polygon_mask).float().to(device)
    proposal_inds = torch.from_numpy(proposal_inds).long().to(device)
    vertex_target = torch.stack(vertex_target).to(device)
    edge_target = torch.from_numpy(edge_target).float().to(device)
    offset_target = torch.from_numpy(offset_target).float().to(device)

    polygon_target = xy_to_class(polygon_target, polygon_size)
    polygon_target = polygon_target.long()

    return proposal_inds, polygon_target, polygon_mask, vertex_target, polygon_vertex_target, edge_target, offset_target
