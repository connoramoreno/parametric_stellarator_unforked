from pymoab import core, types
import read_vmec
import numpy as np


def rxn_rate(s):
    """Calculates fusion reaction rate in plasma.

    Arguments:
        s (float): closed magnetic flux surface index in range of 0 (magnetic
            axis) to 1 (plasma edge).

    Returns:
        rr (float): fusion reaction rate (1/cm^3/s). Equates to neutron source
            density.
    """
    if s == 1:
        rr = 0
    else:
        # Temperature
        T = 11.5*(1 - s)
        # Ion density
        n = 4.8e20*(1 - s**5)
        # Reaction rate
        rr = 3.68e-18*(n**2)/4*T**(-2/3)*np.exp(-19.94*T**(-1/3))

    return rr


def source_strength(verts, verts_s, id0, id1, id2, id3):
    """Computes neutron source strength for a tetrahedron using five-node
    Gaussian quadrature.

    Arguments:
        verts (list of list of float): list of 3D Cartesian coordinates of each
            vertex in form [x (cm), y (cm), z (cm)].
        verts_s (list of float): list of closed flux surface indices for each
            vertex.
        id0 (int): first tetrahedron index in MOAB canonical numbering system.
        id1 (int): second tetrahedron index in MOAB canonical numbering system.
        id2 (int): third tetrahedron index in MOAB canonical numbering system.
        id3 (int): fourth tetrahedron index in MOAB canonical numbering system.

    Returns:
        ss (float): integrated source strength for tetrahedron.
    """
    # Define vertices for tetrahedron
    verts0 = verts[id0]
    verts1 = verts[id1]
    verts2 = verts[id2]
    verts3 = verts[id3]

    # Compute fusion source density at each vertex
    ss0 = rxn_rate(verts_s[id0])
    ss1 = rxn_rate(verts_s[id1])
    ss2 = rxn_rate(verts_s[id2])
    ss3 = rxn_rate(verts_s[id3])

    # Define barycentric coordinates for integration points
    bary_coords = [
        [0.25, 0.25, 0.25, 0.25],
        [0.5, 1/6, 1/6, 1/6],
        [1/6, 0.5, 1/6, 1/6],
        [1/6, 1/6, 0.5, 1/6],
        [1/6, 1/6, 1/6, 0.5]
    ]

    # Define weights for integration points
    int_w = [-0.8, 0.45, 0.45, 0.45, 0.45, 0.45]
    
    # Interpolate source strength at integration points
    ss_int_pts = []
    for pt in bary_coords:
        ss_int = pt[0]*ss0 + pt[1]*ss1 + pt[2]*ss2 + pt[3]*ss3
        ss_int_pts.append(ss_int)
    
    # Compute graph of tetrahedral vertices
    T = [
        [
            verts0[0] - verts3[0],
            verts1[0] - verts3[0],
            verts2[0] - verts3[0]
        ],
        [
            verts0[1] - verts3[1],
            verts1[1] - verts3[1],
            verts2[1] - verts3[1]
        ],
        [
            verts0[2] - verts3[2],
            verts1[2] - verts3[2],
            verts2[2] - verts3[2]
        ]
    ]
    
    # Compute Jacobian of graph
    Jac = np.linalg.det(T)
    # Compute volume of tetrahedron
    vol = np.abs(Jac)/6
    # Compute source strength of tetrahedron
    ss = vol*sum(i*j for i, j in zip(int_w, ss_int_pts))

    return ss


def create_tet(
    moab_core, tag_handle, mesh_set, moab_verts, verts, verts_s, id0, id1, id2,
    id3):
    """Creates tetrahedron and adds to moab core.

    Arguments:
        moab_core (object): PyMOAB core instance.
        tag_handle (TagHandle): PyMOAB source strength tag.
        mesh_set (EntityHandle): PyMOAB mesh set.
        moab_verts (list of EntityHandle): list of mesh vertices.
        verts (list of list of float): list of 3D Cartesian coordinates of each
            vertex in form [x (cm), y (cm), z (cm)].
        verts_s (list of float): list of closed flux surface indices for each
            vertex.
        id0 (int): first tetrahedron index in MOAB canonical numbering system.
        id1 (int): second tetrahedron index in MOAB canonical numbering system.
        id2 (int): third tetrahedron index in MOAB canonical numbering system.
        id3 (int): fourth tetrahedron index in MOAB canonical numbering system.
    """
    # Define vertices for tetrahedron
    tet_verts = [
            moab_verts[id0],
            moab_verts[id1],
            moab_verts[id2],
            moab_verts[id3]
        ]
    # Create tetrahedron in PyMOAB
    tet = moab_core.create_element(types.MBTET, tet_verts)
    # Add tetrahedron to PyMOAB core instance
    moab_core.add_entity(mesh_set, tet)
    # Compute source strength for tetrahedron
    ss = source_strength(verts, verts_s, id0, id1, id2, id3)
    # Tag tetrahedra with source strength data
    moab_core.tag_set_data(tag_handle, tet, [ss])


def source_mesh(plas_eq, num_s, num_theta, num_phi):
    """Creates H5M volumetric mesh defining fusion source using PyMOAB and
    user-supplied plasma equilibrium VMEC data.

    Arguments:
        plas_eq (str): path to plasma equilibrium NetCDF file.
        num_s (int): number of closed magnetic flux surfaces defining mesh.
        num_theta (int): number of poloidal angles defining mesh.
        num_phi (int): number of toroidal angles defining mesh.
    """
    # Load plasma equilibrium data
    vmec = read_vmec.vmec_data(plas_eq)

    # Generate list for closed magnetic flux surface indices in idealized space
    # to be included in mesh
    s_list = np.linspace(0, 1, num = num_s)
    # Generate list for poloidal angles in idealized space to be included in
    # mesh
    theta_list = np.linspace(0, 2*np.pi, num = num_theta)
    # Generate list for toroidal angles in idealized space to be included in
    # mesh
    phi_list = np.linspace(0, 2*np.pi, num = num_phi)

    # Create PyMOAB core instance
    mbc = core.Core()

    # Define data type for source strength tag
    tag_type = types.MB_TYPE_DOUBLE
    # Define tag size for source strength tag (1 double value)
    tag_size = 1
    # Define storage type for source strength
    storage_type = types.MB_TAG_DENSE
    # Define tag handle for source strength
    tag_handle = mbc.tag_get_handle(
        "SourceStrength", tag_size, tag_type, storage_type,
        create_if_missing = True
    )

    # Initialize list of vertices in mesh
    verts = []
    # Initialize list of closed flux surface indices for each vertex
    verts_s = []

    # Compute vertices in Cartesian space
    for phi in phi_list:
        # Determine vertex at magnetic axis
        verts += [list(vmec.vmec2xyz(s_list[0], theta_list[0], phi))]
        # Store s for vertex
        verts_s += [s_list[0]]
        for s in s_list[1:]:
            for theta in theta_list:
                # Detemine vertices beyond magnetic axis in same toroidal angle
                verts += [list(vmec.vmec2xyz(s, theta, phi))]
                # Store s for vertex
                verts_s += [s]

    # Create vertices in PyMOAB
    mbc_verts = mbc.create_vertices(verts)

    tet_set = mbc.create_meshset()
    mbc.add_entity(tet_set, mbc_verts)

    # Create tetrahedra, looping through vertices
    for i, phi in enumerate(phi_list[:-1]):
        # Define index for magnetic axis at phi
        ma_id = i*((num_s - 1)*num_theta + 1)
        # Define index for magnetic axis at next phi
        next_ma_id = ma_id + (num_s - 1)*num_theta + 1

        # Create tetrahedra for wedges at center of plasma
        for k, theta in enumerate(theta_list[:-1], 1):
            # Define indices for wedges at center of plasma
            wedge_id0 = ma_id
            wedge_id1 = ma_id + k
            wedge_id2 = ma_id + k + 1
            wedge_id3 = next_ma_id
            wedge_id4 = next_ma_id + k
            wedge_id5 = next_ma_id + k + 1

            # Define three tetrahedra for wedge
            create_tet(
                mbc, tag_handle, tet_set, mbc_verts, verts, verts_s, wedge_id1,
                wedge_id2, wedge_id4, wedge_id0
            )
            create_tet(
                mbc, tag_handle, tet_set, mbc_verts, verts, verts_s, wedge_id5,
                wedge_id4, wedge_id2, wedge_id3
            )
            create_tet(
                mbc, tag_handle, tet_set, mbc_verts, verts, verts_s, wedge_id0,
                wedge_id2, wedge_id4, wedge_id3
            )

        # Create tetrahedra for hexahedra beyond center of plasma
        for j, s in enumerate(s_list[1:-1]):
            # Define index at current closed magnetic flux surface
            s_offset = j*num_theta
            # Define index at next closed magnetic flux surface
            next_s_offset = s_offset + num_theta

            # Create tetrahedra for current hexahedron
            for k, theta in enumerate(theta_list[:-1], 1):
                # Define indices for hexahedron beyond center of plasma
                tet_id0 = ma_id + s_offset + k
                tet_id1 = ma_id + next_s_offset + k
                tet_id2 = ma_id + next_s_offset + k + 1
                tet_id3 = ma_id + s_offset + k + 1
                tet_id4 = next_ma_id + s_offset + k
                tet_id5 = next_ma_id + next_s_offset + k
                tet_id6 = next_ma_id + next_s_offset + k + 1
                tet_id7 = next_ma_id + s_offset + k + 1

                # Define five tetrahedra for hexahedron
                create_tet(
                    mbc, tag_handle, tet_set, mbc_verts, verts, verts_s,
                    tet_id0, tet_id3, tet_id1, tet_id4
                )
                create_tet(
                    mbc, tag_handle, tet_set, mbc_verts, verts, verts_s,
                    tet_id7, tet_id4, tet_id6, tet_id3
                )
                create_tet(
                    mbc, tag_handle, tet_set, mbc_verts, verts, verts_s,
                    tet_id2, tet_id1, tet_id3, tet_id6
                )
                create_tet(
                    mbc, tag_handle, tet_set, mbc_verts, verts, verts_s,
                    tet_id5, tet_id6, tet_id4, tet_id1
                )
                create_tet(
                    mbc, tag_handle, tet_set, mbc_verts, verts, verts_s,
                    tet_id3, tet_id1, tet_id4, tet_id6
                )

    # Export mesh
    mbc.write_file("SourceMesh.h5m")
