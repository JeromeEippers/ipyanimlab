import copy

import numpy as np

try :
    import fbx
    import fbx.FbxCommon as fb
    _fbx_available_ = True
except ImportError:
    _fbx_available_ = False

from .render.material import Material



def load_fbx_scene(path):
    lSdkManager, lScene = fb.InitializeSdkObjects()
    status = fb.LoadScene(lSdkManager, lScene, path)
    if not status:
        raise Exception('error parsing fbx file')
    return lScene


def read_skeleton(fbx_scene):
    bone_names = []
    bone_parents = []
    bone_matrices = np.zeros([1024, 4, 4], dtype=np.float32)

    def _skel(node, parent_id):

        bone_names.append(node.GetName())
        bone_parents.append(parent_id)

        m = node.EvaluateLocalTransform()
        bone_id = len(bone_names)-1
        for i in range(4):
            for j in range(4):
                bone_matrices[bone_id, j, i] = m.Get(i, j) #column major vs row major

        for i in range(node.GetChildCount()):
            childnode = node.GetChild(i)
            if isinstance(childnode.GetNodeAttribute(), fbx.FbxMesh) == False:
                _skel(childnode, bone_id)

    root_node = fbx_scene.GetRootNode()
    _skel(root_node.GetChild(0), -1)

    bone_matrices = bone_matrices[:len(bone_names), :, :]

    return bone_names, bone_matrices, np.asarray(bone_parents, dtype=np.int32)


def find_mesh_node(pScene):
    def _get_mesh(pNode):
        if isinstance(pNode.GetNodeAttribute(), fbx.FbxMesh):
            return pNode
        for i in range(pNode.GetChildCount()):
            ret = _get_mesh(pNode.GetChild(i))
            if ret:
                return ret

    node = _get_mesh(pScene.GetRootNode())
    if node :
        return node
    return None


def read_vertex_buffer(lMeshNode):
    lMesh = lMeshNode.GetNodeAttribute()
    lControlPointsCount = lMesh.GetControlPointsCount()
    lControlPoints = lMesh.GetControlPoints()

    m = lMeshNode.EvaluateGlobalTransform()

    # 3pos, 3normal, 3vertex color, 1ao
    vertexstride = 10
    vertices = np.ones((lControlPointsCount, vertexstride), dtype=np.float32)

    for i in range(lControlPointsCount):
        # get positions
        vertices[i, :3] = list(m.MultT(lControlPoints[i]))[:3]

        # get normals
        for j in range(lMesh.GetLayerCount()):
            leNormals = lMesh.GetLayer(j).GetNormals()
            if leNormals:
                if leNormals.GetMappingMode() == fbx.FbxLayerElement.eByControlPoint:
                    if leNormals.GetReferenceMode() == fbx.FbxLayerElement.eDirect:
                        vertices[i, 3:6] = list(m.MultT(leNormals.GetDirectArray().GetAt(i)))[:3]
            vcolors = lMesh.GetLayer(j).GetVertexColors()
            if vcolors:
                if vcolors.GetMappingMode() == fbx.FbxLayerElement.eByControlPoint:
                    if vcolors.GetReferenceMode() == fbx.FbxLayerElement.eDirect:
                        vertices[i, 6:9] = list(vcolors.GetDirectArray().GetAt(i))[:3]

    for j in range(lMesh.GetLayerCount()):
        vertex_id = 0
        vcolors = lMesh.GetLayer(j).GetVertexColors()
        if vcolors:
            if vcolors.GetMappingMode() == fbx.FbxLayerElement.eByPolygonVertex:
                if vcolors.GetReferenceMode() == fbx.FbxLayerElement.eIndexToDirect:
                    lPolygonCount = lMesh.GetPolygonCount()
                    for k in range(lPolygonCount):
                        lPolygonSize = lMesh.GetPolygonSize(k)
                        for l in range(lPolygonSize):
                            lControlPointIndex = lMesh.GetPolygonVertex(k, l)
                            index = vcolors.GetIndexArray().GetAt(vertex_id)
                            col = vcolors.GetDirectArray().GetAt(index)
                            vertices[lControlPointIndex, 6:9] = [col.mRed, col.mGreen, col.mBlue]
                            vertex_id += 1

    return vertices


def read_index_buffer(lMeshNode, read_material_color=True):
    lMesh = lMeshNode.GetNodeAttribute()
    lPolygonCount = lMesh.GetPolygonCount()
    materialMapping = None

    materials = []
    if read_material_color:
        materialMapping = lMesh.GetElementMaterial(0).GetIndexArray()
        for i in range(lMeshNode.GetMaterialCount()):
            name = lMeshNode.GetMaterial(i).GetName()
            color = np.zeros([3], dtype=np.float32)
            color[0] = lMeshNode.GetMaterial(i).Diffuse.Get()[0]
            color[1] = lMeshNode.GetMaterial(i).Diffuse.Get()[1]
            color[2] = lMeshNode.GetMaterial(i).Diffuse.Get()[2]
            material = Material(name, color)
            materials.append([material, np.zeros(lPolygonCount * 10, dtype=np.int32), 0])
    else:
        material = Material('default', np.ones(3, dtype=np.float32))
        materials.append([material, np.zeros(lPolygonCount * 10, dtype=np.int32), 0])

    for i in range(lPolygonCount):
        lPolygonSize = lMesh.GetPolygonSize(i)

        material_i = 0
        if read_material_color:
            material_i = materialMapping.GetAt(i)

        # retriangulate
        for j in range(2, lPolygonSize):
            materials[material_i][1][materials[material_i][2]] = lMesh.GetPolygonVertex(i, 0)
            materials[material_i][2] += 1

            materials[material_i][1][materials[material_i][2]] = lMesh.GetPolygonVertex(i, j - 1)
            materials[material_i][2] += 1

            materials[material_i][1][materials[material_i][2]] = lMesh.GetPolygonVertex(i, j)
            materials[material_i][2] += 1

    # combine all the indices and fill the materials
    output_materials = []
    indices = np.zeros(lPolygonCount * 10, dtype=np.int32)
    first_index = 0
    for mat in materials:
        mat[0].first_index = first_index
        mat[0].index_count = mat[2]

        indices[mat[0].first_index:mat[0].first_index + mat[0].index_count] = mat[1][:mat[0].index_count]
        output_materials.append(mat[0])

        first_index += mat[2]

    return indices[:output_materials[-1].first_index + output_materials[-1].index_count], output_materials


def read_bindpose(lMeshNode, bone_names):
    lMesh = lMeshNode.GetNodeAttribute()
    skin = lMesh.GetDeformer(0,fbx.FbxDeformer.eSkin)
    clustercount = skin.GetClusterCount()

    bindpose = np.zeros([len(bone_names), 4, 4], dtype=np.float32)

    for clusterid in range(clustercount):
        cluster = skin.GetCluster(clusterid)
        linkedNode = cluster.GetLink()

        boneid = bone_names.index(linkedNode.GetName())
        if boneid < 0:
            raise Exception('bone {} not found in skeleton'.format(linkedNode.GetName()))

        m = fbx.FbxAMatrix()
        m = cluster.GetTransformLinkMatrix(m)
        m = m.Inverse()
        for i in range(4):
            for j in range(4):
                bindpose[boneid, j, i] = m.Get(i, j) # column major vs row major

    return bindpose


def read_skinning(lMeshNode, bone_names):
    lMesh = lMeshNode.GetNodeAttribute()
    lControlPointsCount = lMesh.GetControlPointsCount()
    weights = np.zeros([lControlPointsCount, 8], dtype=np.float32)
    indices = np.zeros([lControlPointsCount, 8], dtype=np.int32)
    counts = np.zeros([lControlPointsCount], dtype=np.int32)

    skin = lMesh.GetDeformer(0, fbx.FbxDeformer.eSkin)
    clustercount = skin.GetClusterCount()

    for clusterid in range(clustercount):
        cluster = skin.GetCluster(clusterid)
        linkedNode = cluster.GetLink()

        boneid = bone_names.index(linkedNode.GetName())
        if boneid < 0:
            raise Exception('bone {} not found in skeleton'.format(linkedNode.GetName()))

        vertcount = cluster.GetControlPointIndicesCount()
        for k in range(vertcount):
            vertindex = cluster.GetControlPointIndices()[k]
            index = counts[vertindex]
            indices[vertindex, index] = boneid
            weights[vertindex, index] = cluster.GetControlPointWeights()[k]
            counts[vertindex] += 1


    ind = np.argsort(weights)[:,-4:]
    normalizeweights = np.zeros([lControlPointsCount, 4], dtype=np.float32)
    normalizeindices = np.zeros([lControlPointsCount, 4], dtype=np.int32)

    for i in range(lControlPointsCount):
        normalizeweights[i,:] = weights[i,ind[i]]
        normalizeweights[i, :] /= np.sum(normalizeweights[i, :])
        normalizeindices[i, :] = indices[i, ind[i]]

    return normalizeindices, normalizeweights


def import_fbx_asset(viewer, path, read_material):
    if _fbx_available_ == False:
        raise Exception('Missing an Fbx library')

    scene = load_fbx_scene(path)

    mesh_node = find_mesh_node(scene)
    vbuffer = read_vertex_buffer(mesh_node)
    ibuffer, materials = read_index_buffer(mesh_node, read_material_color=read_material)

    names, restXforms, parents = read_skeleton(scene)

    sibuffer = None 
    swbuffer = None
    bindXforms = None

    if len(names)>1 :
        
        sibuffer, swbuffer = read_skinning(mesh_node, names)
        bindXforms = read_bindpose(mesh_node, names)
        sibuffer = sibuffer.flatten()  
        swbuffer = swbuffer.flatten() 

    else:
        names = None 
        restXforms = None 
        parents = None
        
    return viewer.create_asset(vbuffer.flatten(), ibuffer.flatten(), materials, sibuffer, swbuffer, bindXforms, restXforms, names, parents, path)