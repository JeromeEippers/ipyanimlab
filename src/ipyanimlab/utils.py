import numpy as np
from copy import deepcopy, copy


"""
Most of the functions from this file are comming from the LAFAN1 code,
with a few additions like the matrices functions and the 'pq' functions

@article{harvey2020robust,
author    = {Félix G. Harvey and Mike Yurick and Derek Nowrouzezahrai and Christopher Pal},
title     = {Robust Motion In-Betweening},
booktitle = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH)},
publisher = {ACM},
volume    = {39},
number    = {4},
year      = {2020}
}
"""

def length(x, axis=-1, keepdims=True):
    """
    Computes vector norm along a tensor axis(axes)

    :param x: tensor
    :param axis: axis(axes) along which to compute the norm
    :param keepdims: indicates if the dimension(s) on axis should be kept
    :return: The length or vector of lengths.
    """
    lgth = np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims))
    return lgth


def normalize(x, axis=-1, eps=1e-8):
    """
    Normalizes a tensor over some axis (axes)

    :param x: data tensor
    :param axis: axis(axes) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    res = x / (length(x, axis=axis) + eps)
    return res


def quat_normalize(x, eps=1e-8):
    """
    Normalizes a quaternion tensor

    :param x: data tensor
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized quaternions tensor
    """
    res = normalize(x, eps=eps)
    return res


def angle_axis_to_quat(angle, axis):
    """
    Converts from and angle-axis representation to a quaternion representation

    :param angle: angles tensor
    :param axis: axis tensor
    :return: quaternion tensor
    """
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q


def euler_to_quat(e, order='zyx'):
    """

    Converts from an euler representation to a quaternion representation

    :param e: euler tensor
    :param order: order of euler rotations
    :return: quaternion tensor
    """
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = angle_axis_to_quat(e[..., 0], axis[order[0]])
    q1 = angle_axis_to_quat(e[..., 1], axis[order[1]])
    q2 = angle_axis_to_quat(e[..., 2], axis[order[2]])

    return quat_mul(q0, quat_mul(q1, q2))


def quat_to_mat(rot, pos):
    """
    Converts from a quaternion representation to a matrix 4x4 representation
    :param rot: quaternion tensor
    :param pos: position tensor
    :return: matrix4x4 tensor
    """
    matrices = np.eye(4, dtype=np.float32) * np.ones_like(rot[..., :1, np.newaxis].repeat(4, axis=-1), dtype=np.float32)

    x = quat_mul_vec(rot, np.asarray([1.0, 0.0, 0.0]))
    y = quat_mul_vec(rot, np.asarray([0.0, 1.0, 0.0]))
    z = quat_mul_vec(rot, np.asarray([0.0, 0.0, 1.0]))

    matrices[..., :3, 0] = x
    matrices[..., :3, 1] = y
    matrices[..., :3, 2] = z
    matrices[..., :3, 3] = pos[..., :3]
    return matrices


def quat_inv(q):
    """
    Inverts a tensor of quaternions

    :param q: quaternion tensor
    :return: tensor of inverted quaternions
    """
    res = np.asarray([1, -1, -1, -1], dtype=np.float32) * q
    return res


def quat_fk(lrot, lpos, parents):
    """
    Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations

    :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], lpos[..., i:i+1, :]) + gp[parents[i]])
        gr.append(quat_mul    (gr[parents[i]], lrot[..., i:i+1, :]))

    res = np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)
    return res


def quat_ik(grot, gpos, parents):
    """
    Performs Inverse Kinematics (IK) on global quaternions and global positions to retrieve local representations

    :param grot: tensor of global quaternions with shape (..., Nb of joints, 4)
    :param gpos: tensor of global positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of local quaternion, local positions
    """
    res = [
        np.concatenate([
            grot[..., :1, :],
            quat_mul(quat_inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
        ], axis=-2),
        np.concatenate([
            gpos[..., :1, :],
            quat_mul_vec(
                quat_inv(grot[..., parents[1:], :]),
                gpos[..., 1:, :] - gpos[..., parents[1:], :]),
        ], axis=-2)
    ]

    return res


def quat_mul(x, y, short_path=False):
    """
    Performs quaternion multiplication on arrays of quaternions

    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    if short_path:
        len = np.sum(x * y, axis=-1)
        neg = len < 0.0
        y[neg] = -y[neg]

    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

    return res


def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).

    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    t = 2.0 * np.cross(q[..., 1:], x)
    res = x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)

    return res


def quat_slerp(x, y, a):
    """
    Performs spherical linear interpolation (SLERP) between x and y, with proportion a

    :param x: quaternion tensor
    :param y: quaternion tensor
    :param a: indicator (between 0 and 1) of completion of the interpolation.
    :return: tensor of interpolation results
    """
    len = np.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = np.zeros_like(x[..., 0]) + a
    amount0 = np.zeros(a.shape)
    amount1 = np.zeros(a.shape)

    linear = (1.0 - len) < 0.01
    omegas = np.arccos(len[~linear])
    sinoms = np.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = np.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = np.sin(a[~linear] * omegas) / sinoms
    res = amount0[..., np.newaxis] * x + amount1[..., np.newaxis] * y

    return res


def vec_lerp(x, y, a):
    """
    Performs linear interpolation (LERP) between x and y, with proportion a

    :param x: vector tensor
    :param y: vector tensor
    :param a: indicator (between 0 and 1) of completion of the interpolation.
    :return: tensor of interpolation results
    """
    a = np.zeros_like(x) + a
    return (1.0-a) * x + a * y


def quat_between(x, y, up=None):
    """
    Quaternion rotations between two 3D-vector arrays

    :param x: tensor of 3D vectors
    :param y: tensor of 3D vectors
    :param up: tensor of 3D vectors (not mandatory)
    :return: tensor of quaternions
    """
    if up is not None:
        right = np.cross(x, up)
        ortho_up = normalize(np.cross(right, x))
        projected_on_plane = normalize(y - ortho_up * np.sum(ortho_up * y, axis=-1)[..., np.newaxis])

        return quat_mul(quat_between(projected_on_plane, y), quat_between(x, projected_on_plane))

    res = np.concatenate([
        np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., np.newaxis] +
        np.sum(x * y, axis=-1)[..., np.newaxis],
        np.cross(x, y)], axis=-1)
    return quat_normalize(res)


def m3x3_to_quat(ts, eps=1e-10):
    """
    convert matrix3x3 tensor into a quaternion tensor

    :param ts: matrix tensor [..., 3, 3] actually works with a [...,4,4]
    :return: tensor of quaternions
    """
    qs = np.empty_like(ts[..., :1, 0].repeat(4, axis=-1))

    t = ts[..., 0, 0] + ts[..., 1, 1] + ts[..., 2, 2]

    s = 0.5 / np.sqrt(np.maximum(t + 1, eps))
    qs = np.where((t > 0)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        (0.25 / s)[..., np.newaxis],
        (s * (ts[..., 2, 1] - ts[..., 1, 2]))[..., np.newaxis],
        (s * (ts[..., 0, 2] - ts[..., 2, 0]))[..., np.newaxis],
        (s * (ts[..., 1, 0] - ts[..., 0, 1]))[..., np.newaxis]
    ], axis=-1), qs)

    c0 = (ts[..., 0, 0] > ts[..., 1, 1]) & (ts[..., 0, 0] > ts[..., 2, 2])
    s0 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 0, 0] - ts[..., 1, 1] - ts[..., 2, 2], eps))
    qs = np.where(((t <= 0) & c0)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 2, 1] - ts[..., 1, 2]) / s0)[..., np.newaxis],
        (s0 * 0.25)[..., np.newaxis],
        ((ts[..., 0, 1] + ts[..., 1, 0]) / s0)[..., np.newaxis],
        ((ts[..., 0, 2] + ts[..., 2, 0]) / s0)[..., np.newaxis]
    ], axis=-1), qs)

    c1 = (~c0) & (ts[..., 1, 1] > ts[..., 2, 2])
    s1 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 1, 1] - ts[..., 0, 0] - ts[..., 2, 2], eps))
    qs = np.where(((t <= 0) & c1)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 0, 2] - ts[..., 2, 0]) / s1)[..., np.newaxis],
        ((ts[..., 0, 1] + ts[..., 1, 0]) / s1)[..., np.newaxis],
        (s1 * 0.25)[..., np.newaxis],
        ((ts[..., 1, 2] + ts[..., 2, 1]) / s1)[..., np.newaxis]
    ], axis=-1), qs)

    c2 = (~c0) & (~c1)
    s2 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 2, 2] - ts[..., 0, 0] - ts[..., 1, 1], eps))
    qs = np.where(((t <= 0) & c2)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 1, 0] - ts[..., 0, 1]) / s2)[..., np.newaxis],
        ((ts[..., 0, 2] + ts[..., 2, 0]) / s2)[..., np.newaxis],
        ((ts[..., 1, 2] + ts[..., 2, 1]) / s2)[..., np.newaxis],
        (s2 * 0.25)[..., np.newaxis]
    ], axis=-1), qs)

    return qs


def m4x4_to_qp(ts, eps=1e-10):
    """
    convert a matrix4x4 tensor in quat pos tensors

    :param ts: tensor of matrix 4x4
    :return: tuple of tensors of quaternions and positions
    """
    return (m3x3_to_quat(ts), ts[:, :, :3, 3])


def qp_mul(qp_a, qp_b):
    """
    Multiply two quatpos (similar to matrix multiplication in numpy)

    :param qp_a: tuple of tensors of quaternions and positions
    :param qp_b: tuple of tensors of quaternions and positions
    :return: tuple of tensors of quaternions and positions
    """
    return (
        quat_mul(qp_a[0], qp_b[0]),
        quat_mul_vec(qp_a[0], qp_b[1]) + qp_a[1]
    )


def qp_inv(qp):
    """
    inverse a quatpos

    :param qp: tuple of tensors of quaternions and positions
    :return: tuple of tensors of quaternions and positions
    """
    qs = quat_inv(qp[0])
    ps = np.asarray([-1,-1,-1], dtype=np.float32) * qp[1]
    return qs, quat_mul_vec(qs, ps)


def interpolate_local(lcl_r_mb, lcl_q_mb, n_past, n_future):
    """
    Performs interpolation between 2 frames of an animation sequence.

    The 2 frames are indirectly specified through n_past and n_future.
    SLERP is performed on the quaternions
    LERP is performed on the root's positions.

    :param lcl_r_mb:  Local/Global root positions (B, T, 1, 3)
    :param lcl_q_mb:  Local quaternions (B, T, J, 4)
    :param n_past:    Number of frames of past context
    :param n_future:  Number of frames of future context
    :return: Interpolated root and quats
    """
    # Extract last past frame and target frame
    start_lcl_r_mb = lcl_r_mb[:, n_past - 1, :, :][:, None, :, :]  # (B, 1, J, 3)
    end_lcl_r_mb = lcl_r_mb[:, -n_future, :, :][:, None, :, :]

    start_lcl_q_mb = lcl_q_mb[:, n_past - 1, :, :]
    end_lcl_q_mb = lcl_q_mb[:, -n_future, :, :]

    # LERP Local Positions:
    n_trans = lcl_r_mb.shape[1] - (n_past + n_future)
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    offset = end_lcl_r_mb - start_lcl_r_mb

    const_trans    = np.tile(start_lcl_r_mb, [1, n_trans + 2, 1, 1])
    inter_lcl_r_mb = const_trans + (interp_ws)[None, :, None, None] * offset

    # SLERP Local Quats:
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    inter_lcl_q_mb = np.stack(
        [(quat_normalize(quat_slerp(quat_normalize(start_lcl_q_mb), quat_normalize(end_lcl_q_mb), w))) for w in
         interp_ws], axis=1)

    return inter_lcl_r_mb, inter_lcl_q_mb


def remove_quat_discontinuities(rotations):
    """

    Removing quat discontinuities on the time dimension (removing flips)

    :param rotations: Array of quaternions of shape (T, J, 4)
    :return: The processed array without quaternion inversion.
    """
    rots_inv = -rotations

    for i in range(1, rotations.shape[0]):
        # Compare dot products
        replace_mask = np.sum(rotations[i - 1: i] * rotations[i: i + 1], axis=-1) < np.sum(
            rotations[i - 1: i] * rots_inv[i: i + 1], axis=-1)
        replace_mask = replace_mask[..., np.newaxis]
        rotations[i] = replace_mask * rots_inv[i] + (1.0 - replace_mask) * rotations[i]

    return rotations


# Orient the data according to the las past keframe
def rotate_at_frame(X, Q, parents, n_past=10):
    """
    Re-orients the animation data according to the last frame of past context.

    :param X: tensor of local positions of shape (Batchsize, Timesteps, Joints, 3)
    :param Q: tensor of local quaternions (Batchsize, Timesteps, Joints, 4)
    :param parents: list of parents' indices
    :param n_past: number of frames in the past context
    :return: The rotated positions X and quaternions Q
    """
    # Get global quats and global poses (FK)
    global_q, global_x = quat_fk(Q, X, parents)

    key_glob_Q = global_q[:, n_past - 1: n_past, 0:1, :]  # (B, 1, 1, 4)
    forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] \
                 * quat_mul_vec(key_glob_Q, np.array([0, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :])
    forward = normalize(forward)
    yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))
    new_glob_Q = quat_mul(quat_inv(yrot), global_q)
    new_glob_X = quat_mul_vec(quat_inv(yrot), global_x)

    # back to local quat-pos
    Q, X = quat_ik(new_glob_Q, new_glob_X, parents)

    return X, Q


def extract_feet_contacts(pos, lfoot_idx, rfoot_idx, velfactor=0.02):
    """
    Extracts binary tensors of feet contacts

    :param pos: tensor of global positions of shape (Timesteps, Joints, 3)
    :param lfoot_idx: indices list of left foot joints
    :param rfoot_idx: indices list of right foot joints
    :param velfactor: velocity threshold to consider a joint moving or not
    :return: binary tensors of left foot contacts and right foot contacts
    """
    lfoot_xyz = (pos[1:, lfoot_idx, :] - pos[:-1, lfoot_idx, :]) ** 2
    contacts_l = (np.sum(lfoot_xyz, axis=-1) < velfactor)

    rfoot_xyz = (pos[1:, rfoot_idx, :] - pos[:-1, rfoot_idx, :]) ** 2
    contacts_r = (np.sum(rfoot_xyz, axis=-1) < velfactor)

    # Duplicate the last frame for shape consistency
    contacts_l = np.concatenate([contacts_l, contacts_l[-1:]], axis=0)
    contacts_r = np.concatenate([contacts_r, contacts_r[-1:]], axis=0)

    return contacts_l, contacts_r


def mirror(quats, pos, parents, names, left_name="Left", right_name="Right"):
    """Mirror an animation

    Args:
        :quats: (local quaternion tensor): [frame_count, bone_count, 4]
        :pos: (local translation tensor): [frame_count, bone_count, 3]
        :parents: (int tensor): the parent index in the skeleton
        :names: (list of strings): the names of the bones
        :left_name: (str, optional): the name to search for in the replace. Defaults to "Left".
        :right_name: (str, optional): the name to search for in the replace. Defaults to "Right".

    Returns:
        :tuple of tensor: the new quats and pos tensors
    """
    mirror_index = np.array([names.index(n.replace(left_name, 'TTT').replace(right_name, left_name).replace('TTT',right_name)) for n in names])

    mirror_quats, mirror_pos = quat_fk(quats, pos, parents)
    mirror_quats = mirror_quats[:, mirror_index, :]
    mirror_pos = mirror_pos[:, mirror_index, :]

    mirror_pos[:, :, 0] *= -1
    mirror_quats[:, :, 2] *= -1
    mirror_quats[:, :, 3] *= -1

    mirror_quats = quat_mul(mirror_quats, euler_to_quat(np.array([0, np.pi, 0]))[np.newaxis,:])

    return quat_ik(mirror_quats, mirror_pos, parents)


def limb_ik(quats, pos, parents, names, target_quats, target_pos, effector_names=['LeftFoot', 'RightFoot']):
    """compute limb ik giving the end effectors

    Args:
        :quats: (local quaternion tensor): [frame_count, bone_count, 4]
        :pos: (local translation tensor): [frame_count, bone_count, 3]
        :parents: (int tensor): the parent index in the skeleton
        :names: (list of strings): the names of the bones
        :target_quats: (global tensor of quaternions): (framecount, effector_count, 4)
        :target_pos: (global tensor of positions): (framecount, effector_count, 3)
        :effector_names: (string list): the names of the end of the chains

    Returns:
        tupple of tensors (quat pos)
    """
    quats = deepcopy(quats)
    pos = deepcopy(pos)

    frame_count = quats.shape[0]
    
    def _compute_ik(quats, pos, parents, target_id, top, middle, end):
        gq, gp = quat_fk(quats, pos, parents)

        # prepare sizes
        len1 = pos[0, middle, 0]
        len2 = pos[0, end, 0]
        l1l2_2 = np.array([len1 * len2 *2.0], dtype=np.float32).repeat(frame_count, axis=0)
        l1_l2 = np.array([len1*len1 + len2*len2], dtype=np.float32).repeat(frame_count, axis=0)
    
        # compute leg angle
        #------------------------
        ankle_vector = target_pos[:, target_id, :]  - gp[:, top, :]
        ankle_dist = np.sum(ankle_vector*ankle_vector, axis=-1, keepdims=True)[...,0]
        ankle_arccos = (l1_l2 - ankle_dist)/l1l2_2
    
        # apply the rotation on the knee
        _angle = (np.pi - np.arccos(np.maximum(-1.0*np.ones_like(ankle_arccos, dtype=np.float32), ankle_arccos)))/2.0
        quats[:, middle, 0] = np.cos(-_angle)
        quats[:, middle, 3] = np.sin(-_angle)
    
        # recompute the global position of the foot
        quats[...] = quat_normalize(quats)
        gq, gp = quat_fk(quats, pos, parents)
    
        # rotate the leg (must be done on global position, then convert back on local)
        rot = quat_between(gp[:, end, :] - gp[:, top, :], ankle_vector)
        gq[:, top, :] = quat_mul(rot, gq[:, top, :])
        quats[:, top, :] = quat_ik(gq, gp, parents)[0][:, top, :]

        # recompute the global position
        quats[...] = quat_normalize(quats)
        gq, gp = quat_fk(quats, pos, parents)
        
        # force the foot on the constraint in global and convert in local
        gq[:, end, :] = target_quats[:, target_id, :]
        gp[:, end, :] = target_pos[:, target_id, :]
        lq, lp = quat_ik(gq, gp, parents)
        quats[:, end, :] = lq[:, end, :]
        pos[:, end, :] = lp[:, end, :]
    
    effectors = []
    middles = []
    tops = []

    for name in effector_names:
        effector = names.index(name)
        middle = parents[effector]
        top = parents[middle]

        effectors.append(effector)
        middles.append(middle)
        tops.append(top)
    
    _compute_ik(quats, pos, parents, list(range(len(effector_names))), tops, middles, effectors)

    return quats, pos
