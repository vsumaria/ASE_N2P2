import pytest
import numpy as np
from ase.quaternions import Quaternion

TEST_N = 200


def axang_rotm(u, theta):

    u = np.array(u, float)
    u /= np.linalg.norm(u)

    # Cross product matrix for u
    ucpm = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

    # Rotation matrix
    rotm = (np.cos(theta) * np.identity(3) + np.sin(theta) * ucpm +
            (1 - np.cos(theta)) * np.kron(u[:, None], u[None, :]))

    return rotm


def rand_rotm(rng=np.random.RandomState(0)):
    """Axis & angle rotations."""
    u = rng.random(3)
    theta = rng.random() * np.pi * 2

    return axang_rotm(u, theta)


def eulang_rotm(a, b, c, mode='zyz'):

    rota = axang_rotm([0, 0, 1], a)
    rotc = axang_rotm([0, 0, 1], c)

    if mode == 'zyz':
        rotb = axang_rotm([0, 1, 0], b)
    elif mode == 'zxz':
        rotb = axang_rotm([1, 0, 0], b)

    return np.dot(rotc, np.dot(rotb, rota))


@pytest.fixture
def rng():
    return np.random.RandomState(0)


def test_quaternions_rotations(rng):

    # First: test that rotations DO work
    for i in range(TEST_N):
        # n random tests

        rotm = rand_rotm(rng)

        q = Quaternion.from_matrix(rotm)
        assert np.allclose(rotm, q.rotation_matrix())

        # Now test this with a vector
        v = rng.random(3)

        vrotM = np.dot(rotm, v)
        vrotQ = q.rotate(v)

        assert np.allclose(vrotM, vrotQ)


def test_quaternions_gimbal(rng):

    # Second: test the special case of a PI rotation

    rotm = np.identity(3)
    rotm[:2, :2] *= -1               # Rotate PI around z axis

    q = Quaternion.from_matrix(rotm)

    assert not np.isnan(q.q).any()


def test_quaternions_overload(rng):

    # Third: test compound rotations and operator overload
    for i in range(TEST_N):

        rotm1 = rand_rotm(rng)
        rotm2 = rand_rotm(rng)

        q1 = Quaternion.from_matrix(rotm1)
        q2 = Quaternion.from_matrix(rotm2)

        assert np.allclose(np.dot(rotm2, rotm1),
                           (q2 * q1).rotation_matrix())
        # Now test this with a vector
        v = rng.random(3)

        vrotM = np.dot(rotm2, np.dot(rotm1, v))
        vrotQ = (q2 * q1).rotate(v)

        assert np.allclose(vrotM, vrotQ)


def test_quaternions_euler(rng):

    # Fourth: test Euler angles
    for mode in ['zyz', 'zxz']:
        for i in range(TEST_N):

            abc = rng.random(3) * 2 * np.pi

            q_eul = Quaternion.from_euler_angles(*abc, mode=mode)
            rot_eul = eulang_rotm(*abc, mode=mode)

            assert(np.allclose(rot_eul, q_eul.rotation_matrix()))

            # Test conversion back and forth
            abc_2 = q_eul.euler_angles(mode=mode)
            q_eul_2 = Quaternion.from_euler_angles(*abc_2, mode=mode)

            assert(np.allclose(q_eul_2.q, q_eul.q))


def test_quaternions_rotm(rng):

    # Fifth: test that conversion back to rotation matrices works properly
    for i in range(TEST_N):

        rotm1 = rand_rotm(rng)
        rotm2 = rand_rotm(rng)

        q1 = Quaternion.from_matrix(rotm1)
        q2 = Quaternion.from_matrix(rotm2)

        assert(np.allclose(q1.rotation_matrix(), rotm1))
        assert(np.allclose(q2.rotation_matrix(), rotm2))
        assert(np.allclose((q1 * q2).rotation_matrix(), np.dot(rotm1, rotm2)))
        assert(np.allclose((q1 * q2).rotation_matrix(), np.dot(rotm1, rotm2)))


def test_quaternions_axang(rng):

    # Sixth: test conversion to axis + angle
    q = Quaternion()
    n, theta = q.axis_angle()
    assert(theta == 0)

    u = np.array([1, 0.5, 1])
    u /= np.linalg.norm(u)
    alpha = 1.25

    q = Quaternion.from_matrix(axang_rotm(u, alpha))
    n, theta = q.axis_angle()

    assert(np.isclose(theta, alpha))
    assert(np.allclose(u, n))
