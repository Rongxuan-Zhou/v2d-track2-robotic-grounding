"""Tier 2 恢复的纯函数核心：不读任何数据文件，不 import 任何会读数据的模块。

存在的理由：原先 v51 / nopen 从 denoise_v1、denoise_v4、support_bridge import 这些函数，
而 denoise_v1 在 import 时就打开 **Tier 1 的 episodes_metadata.jsonl**，denoise_v4 还会构造 SOMALayer。
提交用的 Tier 2 导出因此在启动时就读过一个 Tier 1 文件——违反档位隔离。
函数体用 ast 从原文件原样抽出（见各段注释），行为不变；export_tier2 的输出须与改前逐位一致。
"""
import numpy as np

import add_metrics as A

M = lambda q: A.quat_wxyz_to_matrix(q)
R_CV = np.diag([1.0, -1.0, -1.0])


# ── 原样抽自 denoise_v1.py ──
def nq(q): return q/(np.linalg.norm(q,axis=-1,keepdims=True)+1e-12)


def qmul(a,b):
    w1,x1,y1,z1=a[...,0],a[...,1],a[...,2],a[...,3]; w2,x2,y2,z2=b[...,0],b[...,1],b[...,2],b[...,3]
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2],axis=-1)


def qinv(q): return np.stack([q[...,0],-q[...,1],-q[...,2],-q[...,3]],axis=-1)


def window_rot(p,W=91):
    q=nq(p[:,3:7]); T=len(q); out=np.zeros(T); h=W//2
    for i in range(T):
        a,b=max(0,i-h),min(T,i+h+1)
        c=np.abs((q[a:b]*q[a:b][:,None,:]).sum(-1)); out[i]=np.degrees(2*np.arccos(np.clip(c.min(),0,1)))
    return out


def window_tr(p,W=91):
    x=p[:,:3]; T=len(x); out=np.zeros(T); h=W//2
    for i in range(T):
        a,b=max(0,i-h),min(T,i+h+1); s=x[a:b]
        out[i]=np.linalg.norm(s[:,None,:]-s[None,:,:],axis=-1).max()
    return out


# ── 原样抽自 denoise_v4.py ──
def mat_to_q(M):
    w=np.sqrt(np.clip(1+M[...,0,0]+M[...,1,1]+M[...,2,2],1e-12,None))/2
    return np.stack([w,(M[...,2,1]-M[...,1,2])/(4*w),(M[...,0,2]-M[...,2,0])/(4*w),(M[...,1,0]-M[...,0,1])/(4*w)],-1)


def to_soma(p):
    q=p.copy(); q[:,:3]=p[:,:3]@R_CV.T
    q[:,3:7]=mat_to_q(np.einsum("ij,tjk->tik",R_CV,A.quat_wxyz_to_matrix(p[:,3:7])))
    return q


def segments(mask,minlen):
    out=[];s=None
    for i,m in enumerate(mask):
        if m and s is None: s=i
        elif not m and s is not None:
            if i-s>=minlen: out.append((s,i))
            s=None
    if s is not None and len(mask)-s>=minlen: out.append((s,len(mask)))
    return out


# ── 原样抽自 support_bridge.py ──
def integrate(p, st):
    """v1 积分器，返回每帧漂移 (DR:(T,3,3), Dx:(T,3))。"""
    q = nq(p[:, 3:7]); x = p[:, :3]; T = len(p)
    dq = np.tile([1., 0, 0, 0], (T, 1)); dx = np.zeros((T, 3))
    for t in range(1, T):
        dq[t] = dq[t-1]; dx[t] = dx[t-1]
        if st[t] and st[t-1]:
            inc = qmul(q[t], qinv(q[t-1]))
            if inc[0] < 0: inc = -inc
            dq[t] = nq(qmul(inc, dq[t-1])); dx[t] = dx[t-1] + (x[t] - x[t-1])
    return M(dq), dx


def apply(p, DR, Dx):
    out = p.copy()
    out[:, 3:7] = mat_to_q(np.einsum("tji,tjk->tik", DR, M(p[:, 3:7])))   # true = D^T obs
    out[:, :3] = p[:, :3] - Dx
    return out


def rot_between(a, b):
    """把单位向量 a 转到 b 的最小旋转。"""
    v = np.cross(a, b); c = float(a @ b); s = np.linalg.norm(v)
    if s < 1e-12: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s**2)


def slerp_R(R0, R1, w):
    from scipy.spatial.transform import Rotation as Rot, Slerp
    return Slerp([0, 1], Rot.from_matrix(np.stack([R0, R1])))(w).as_matrix()


def correct(DR, Dx, segs, target, n, bridge, carry=False):
    """每个静止段求一个常数修正（旋转 C 左乘、平移 c 相加），运动段按 bridge 插值或保持。
    target(s,e) -> (目标 D·n 方向 或 None, 目标 n·Dx 或 None, 目标整体 D 或 None)
    carry=True（候选 B1，计划 agent-doc/plan-tier2-denoise-2026-10-04.md 第 2 版第 4 节）：target 全为 None 的段，不再停在原始积分值，
    改为沿用上一个锚定段 a 的修正。锚点在 a 的中点 m_a 处求得 C_a，即 C_a·DR(m_a) 是那里的漂移估计；按随机游走模型，此后运动段的
    漂移增量期望为 I，所以本段中点 m 的修正取 C = DR(m)·DR(m_a)ᵀ·C_a·DR(m_a)·DR(m)ᵀ（全部增量都观测到时恰好还原）；
    平移修正是加性的，c = c_a。第一个锚定段之前的段照旧。carry=False 时与原实现逐位相同。"""
    DR0 = DR if not carry else DR.copy()
    DR = DR.copy(); Dx = Dx.copy(); T = len(DR)
    fixed = []
    last = None   # 上一个锚定段：(C_a, c_a, m_a)，m_a 是它的中点
    for s, e in segs:
        z_img, h_d, full = target(s, e)
        m = (s + e) // 2
        if full is not None:
            C = full[0] @ DR[m].T; c = full[1] - Dx[m]
        elif carry and z_img is None and h_d is None and last is not None:
            Ca, ca, ma = last
            C = DR0[m] @ DR0[ma].T @ Ca @ DR0[ma] @ DR0[m].T; c = ca.copy()
        else:
            C = np.eye(3) if z_img is None else rot_between(DR[m] @ n, z_img)
            c = np.zeros(3) if h_d is None else (h_d - n @ Dx[m]) * n
        if carry and (full is not None or z_img is not None or h_d is not None):
            last = (C, c, m)
        DR[s:e] = np.einsum("ij,tjk->tik", C, DR[s:e]); Dx[s:e] += c
        fixed.append((s, e))
    # 运动段：两侧都有静止段时按 bridge 插值，否则保持邻近值
    for (s0, e0), (s1, e1) in zip(fixed, fixed[1:]):
        a, b = e0 - 1, s1
        if b - a <= 1: continue
        idx = np.arange(a + 1, b)
        if bridge:
            w = (idx - a) / (b - a)
            DR[idx] = slerp_R(DR[a], DR[b], w); Dx[idx] = (1 - w)[:, None] * Dx[a] + w[:, None] * Dx[b]
        else:
            DR[idx] = DR[a]; Dx[idx] = Dx[a]
    if fixed:
        DR[fixed[-1][1]:] = DR[fixed[-1][1] - 1]; Dx[fixed[-1][1]:] = Dx[fixed[-1][1] - 1]
    return DR, Dx
