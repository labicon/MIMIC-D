from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
import numpy as np

def quat_to_rot6d(quat):
    """Convert quaternion to 6D rotation representation.
    Args:
        quat (np.array): quaternion in wxyz format
    Returns:
        np.array: 6D rotation representation
    """
    r = R.from_quat(quat).as_matrix()

    return r[:3, :2].T.flatten()

def rotvec_to_rot6d(rotvec):
    r = R.from_rotvec(rotvec).as_matrix()

    return r[:3, :2].T.flatten()

def rot6d_to_quat(rot6d):
    """Convert 6D rotation representation to quaternion.
    Args:
        rot6d (np.array): 6D rotation representation
    """
    x_raw = rot6d[:3]
    y_raw = rot6d[3:]
    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    print(f"x: {x}, y: {y}, z: {z}")
    quat = R.from_matrix(np.column_stack((x, y, z))).as_quat()
    
    return quat

def rot6d_to_rotvec(rot6d):
    """Convert 6D rotation representation to quaternion.
    Args:
        rot6d (np.array): 6D rotation representation
    """
    x_raw = rot6d[:3]
    y_raw = rot6d[3:]
    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    print(f"x: {x}, y: {y}, z: {z}")
    rotvec = R.from_matrix(np.column_stack((x, y, z))).as_rotvec()
    
    return rotvec

def SE3_log_map(g):
    p, R = g[:3,3], g[:3,:3]
    r = R.as_rotvec()


def hat_map(w):
    return np.array([[0, -w[2], w[1]],
                        [w[2], 0, -w[0]],
                        [-w[1], w[0], 0]])

def vee_map(mat):
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])

def SO3_log_map(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    if theta == 0:
        xi = np.zeros(3,)
    else:
        xi = theta / (2 * np.sin(theta)) * vee_map(R - R.T)

    return xi

def SE3_log_map(g):
    p, R = g[:3,3], g[:3,:3]
    
    psi = SO3_log_map(R)
    psi_norm = np.linalg.norm(psi)
    psi_hat = hat_map(psi)

    if np.isclose(psi_norm, 0):
        A_inv = np.eye(3) - 0.5 * psi_hat + 1 / 12.0 * psi_hat @ psi_hat

    else:
        cot = 1 / np.tan(psi_norm / 2)
        alpha = (psi_norm /2) * cot

        A_inv = np.eye(3) - 0.5 * psi_hat + (1 - alpha)/(psi_norm**2) * psi_hat @ psi_hat

    v = A_inv @ p
    xi = np.zeros(6,)
    xi[:3] = v
    xi[3:] = psi

    return xi

def SE3_exp_map(xi):
    v, omega = xi[:3], xi[3:]

    omega_hat = hat_map(omega)

    xi_hat = np.zeros((4, 4))
    xi_hat[:3, :3] = omega_hat
    xi_hat[:3, 3] = v

    g = expm(xi_hat)

    return g
     


if __name__ == "__main__":
    rpy = np.array([0.1, 0.2, 0.3])
    quat = R.from_euler("xyz", rpy).as_quat()
    print(f"Matrix: {R.from_quat(quat).as_matrix()}")
    print(f"Quaternion: {quat}")

    rot6d = quat_to_rot6d(quat)
    print(f"6D rotation: {rot6d}")

    quat_recon = rot6d_to_quat(rot6d)
    print(f"Quaternion recon: {quat_recon}")