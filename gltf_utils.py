from collections import deque
from enum import Enum, IntEnum, auto
from typing import Callable, Concatenate, Dict, List, Tuple, Union

import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from matplotlib.axes import Axes
from pygltflib import Accessor, Node, GLTF2

# Workaround for issue with pytransform3d's use of np.int, np.float, etc.
# See: https://stackoverflow.com/a/75837322/802203
np.float = float  # noqa: F401
np.int = int  # noqa: F401
np.object = object  # noqa: F401
np.bool = bool  # noqa: F401

import struct

class SamplerInterpolationType(Enum):
    """
    An enumeration representing the interpolation types in a GLTF sampler.

    This enumeration provides a value for each possible interpolation type in a GLTF sampler.
    Currently, only the LINEAR interpolation type is defined.

    Attributes
    ----------
    LINEAR : Enum
        Represents linear interpolation, which creates a straight line between two keyframes.
    """
    LINEAR = auto()

class ChannelTargetPath(Enum):
    """
    An enumeration representing the target paths in a GLTF channel.

    This enumeration provides a value for each possible target path in a GLTF channel.
    Currently, only the 'translation' and 'rotation' target paths are defined.

    Attributes
    ----------
    translation : Enum
        Represents the translation target path, which corresponds to the movement of an object in 3D space.
    rotation : Enum
        Represents the rotation target path, which corresponds to the rotation of an object in 3D space.
    """
    
    translation = auto()
    rotation = auto()

'''Animation channel data structure to combine times, values, interpolation type and path'''
class AnimationChannelData:
    def __init__(self, times: List[np.ndarray], values: List[np.ndarray],
        interpolation: SamplerInterpolationType,
        path: ChannelTargetPath):
        
        self.times = times
        self.values = values
        self.interpolation = interpolation
        self.path = path

def get_accessor_data(gltf: GLTF2, accessor: Accessor) -> List[Tuple[Union[np.int8, np.uint8, np.int16, np.uint16,np.uint32, np.float32]]]:
    """
    Retrieves data from a given accessor in a GLTF2 object.

    This function accesses the buffer view and buffer associated with the given accessor,
    retrieves the data from the buffer via the URI, and unpacks the data from the buffer,
    element by element. The data is returned as a list of tuples, where each tuple represents
    an element in the buffer.

    Parameters
    ----------
    gltf : GLTF2
        The GLTF2 object that contains the accessor.
    accessor : Accessor
        The accessor from which to retrieve the data.

    Returns
    -------
    List[Tuple[Union[np.int8, np.uint8, np.int16, np.uint16,np.uint32, np.float32]]]
        A list of tuples, where each tuple represents an element in the buffer.
        The type of the elements in the tuple can be np.int8, np.uint8, np.int16, np.uint16,
        np.uint32, or np.float32, depending on the component type of data in the buffer.
        Note: If the component count is 1, the list will contain direct data elements instead of tuples.
    """

    class GLTFComponentType(IntEnum):
        BYTE = 5120
        UNSIGNED_BYTE = 5121
        SHORT = 5122
        UNSIGNED_SHORT = 5123
        UNSIGNED_INT = 5125
        FLOAT = 5126

    GLTF_COMPONENTTYPE_SIZES = {
        GLTFComponentType.BYTE: 1,
        GLTFComponentType.UNSIGNED_BYTE: 1,
        GLTFComponentType.SHORT: 2,
        GLTFComponentType.UNSIGNED_SHORT: 2,
        GLTFComponentType.UNSIGNED_INT: 4,
        GLTFComponentType.FLOAT: 4
    }
    GLTF_ACCESSORTYPE_COUNTS = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16
    }
    GLTF_COMPONENT_UNPACK_FORMATS = {
        GLTFComponentType.BYTE: "b",
        GLTFComponentType.UNSIGNED_BYTE: "B",
        GLTFComponentType.SHORT: "h",
        GLTFComponentType.UNSIGNED_SHORT: "H",
        GLTFComponentType.UNSIGNED_INT: "I",
        GLTFComponentType.FLOAT: "f"
    }

    # Access the buffer view and buffer
    bufferView = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[bufferView.buffer]

    # Get the data from the buffer via the URI
    buffer_data = gltf.get_data_from_buffer_uri(buffer.uri)
    result = []

    # Calculate the element stride in bytes
    elem_stride = GLTF_ACCESSORTYPE_COUNTS[accessor.type] * GLTF_COMPONENTTYPE_SIZES[int(accessor.componentType)]

    # Unpack the data from the buffer, element by element
    for i in range(accessor.count):
        index = bufferView.byteOffset + accessor.byteOffset + i * elem_stride
        base64_elem_data = buffer_data[index:index + elem_stride]
        elem_data = struct.unpack(f"<{GLTF_COMPONENT_UNPACK_FORMATS[accessor.componentType] * GLTF_ACCESSORTYPE_COUNTS[accessor.type]}", base64_elem_data)
        
        # If there's it's a single-element type, don't append a tuple but the element itself
        if len(elem_data) == 1:
            result.append(elem_data[0])
        else:
            result.append(elem_data)
    
    return result

def get_animation_data(gltf_data: GLTF2, animation_idx: int = 0) -> AnimationChannelData:
    """
    Retrieves animation data from a given GLTF2 object.

    This function accesses the animation data associated with the given index in the GLTF2 object.
    If no index is provided, it defaults to the first animation (index 0).

    Parameters
    ----------
    gltf_data : GLTF2
        The GLTF2 object that contains the animation data.
    animation_idx : int, optional
        The index of the animation from which to retrieve the data. Defaults to 0.

    Returns
    -------
    Animation
        The animation data associated with the given index. The returned object is of 
        type AnimationChannelData.
    """
    # Initialize an empty dictionary to store the animation data
    animation_data = {}

    # Get the animation object from the GLTF2 data using the provided index
    animation = gltf_data.animations[animation_idx]

    # Iterate over each channel in the animation
    for channel in animation.channels:
        # Get the sampler associated with the current channel
        sampler = animation.samplers[channel.sampler]
        
        # Retrieve the time and value data from the sampler's input accessor
        times = get_accessor_data(gltf_data, gltf_data.accessors[sampler.input])
        values = get_accessor_data(gltf_data, gltf_data.accessors[sampler.output])
        
        # Get the interpolation type from the sampler
        interpolationType = SamplerInterpolationType[sampler.interpolation]
        
        # Get the target path from the channel
        path = ChannelTargetPath[channel.target.path]

        # Package this info in an AnimationChannelData object
        animation_channel_data = AnimationChannelData(times, values, interpolationType, path)
        
        # Add the animation channel data to the animation data dictionary, 
        # with node index as the key
        animation_data[channel.target.node] = animation_channel_data

    return animation_data

def get_keyframe_at(time: float, channel: AnimationChannelData) -> np.ndarray:
    def calculate_interpolation_fraction(t0: float, t1: float, t: float) -> float:
        """
        Calculates the interpolation fraction for a given time.

        This function calculates the fraction of the way that a given time 't' is between two other times 't0' and 't1'.
        If 't0' and 't1' are equal, the function returns 0.0 if 't' is less than or equal to 't0', and 1.0 otherwise.
        If 't' is outside the range 't0' to 't1', it is clamped to the nearest boundary.

        Parameters
        ----------
        t0 : float
            The start time of the interpolation range.
        t1 : float
            The end time of the interpolation range.
        t : float
            The time at which to calculate the interpolation fraction.

        Returns
        -------
        float
            The interpolation fraction, which is a number between 0.0 and 1.0 indicating how far 't' is between 't0' and 't1'.
        """
        
        # Avoid division by zero if t0 and t1 are equal
        if t0 == t1:
            return 0.0 if t <= t0 else 1.0

        # Ensure t0 <= t <= t1
        t = max(t0, min(t, t1))

        # Calculate the interpolation fraction
        fraction = (t - t0) / (t1 - t0)
        return fraction
    
    def linear_sample_rotation(v0: np.ndarray, v1: np.ndarray, frac: float) -> np.ndarray:
        """
        Performs linear interpolation (SLERP) of rotations between two quaternions according to a given interpolation fraction.

        Parameters
        ----------
        v0 : np.ndarray
            The rotation value of the first keyframe, represented as a quaternion in a numpy array of the form [(x, y, z, w)].
        v1 : np.ndarray
            The rotation value of the second keyframe, represented as a quaternion in a numpy array of the form [(x, y, z, w)].
        frac : float
            The fraction of the way between the first and second keyframes that the result should be at.

        Returns
        -------
        np.ndarray
            The rotation value at the interpolation fraction, represented as a quaternion in a numpy array of the form [(w, x, y, z)].
        """
        return pr.quaternion_slerp(np.array([v0[3], v0[0], v0[1], v0[2]]), np.array([v1[3], v1[0], v1[1], v1[2]]), frac)

    def linear_sample_translation(v0: np.ndarray, v1: np.ndarray, frac: float) -> np.ndarray:
        """
        Performs linear interpolation of translations between two vectors according to a given interpolation fraction.

        Parameters
        ----------
        v0 : np.ndarray
            The translation value of the first keyframe, represented as a 3D vector in a numpy array.
        v1 : np.ndarray
            The translation value of the second keyframe, represented as a 3D vector in a numpy array.
        frac : float
            The fraction of the way between the first and second keyframes that the result should be at.

        Returns
        -------
        np.ndarray
            The translation value of the third point, represented as a 3D vector in a numpy array.
        """
        return v0 + frac * (v1 - v0)

    idx = next(i for i, t in enumerate(channel.times) if t >= time)

    t1, t0 = channel.times[idx], channel.times[idx - 1]
    v1, v0 = channel.values[idx], channel.values[idx - 1]
    frac = calculate_interpolation_fraction(t0, t1, time)

    translation = np.array([0., 0., 0., 1.])
    if channel.path == ChannelTargetPath.translation and channel.interpolation == SamplerInterpolationType.LINEAR:
        translation = linear_sample_translation(v0, v1, frac)

    # Express this as a rotation quaternion of the form (x, y, z, w)
    rotation = np.array([0., 0., 0., 1.])
    if channel.path == ChannelTargetPath.rotation and channel.interpolation == SamplerInterpolationType.LINEAR:
        rotation = linear_sample_rotation(v0, v1, frac)

    # Return the composed transformation sampled at the given time
    return pt.transform_from_pq(np.array([translation[0], translation[1], translation[2], 
                                rotation[0], rotation[1], rotation[2], rotation[3]]))

def skeleton_visitor_DFS(gltf: GLTF2, visitor: Callable[Concatenate[GLTF2, int, int, ...], None], root_node: Node = None, **kwargs) -> None:
    """
    Traverses the skeleton of a GLTF2 object in a depth-first manner and applies a visitor function to each node.

    This function performs a depth-first traversal of the skeleton, starting from the root node.
    For each node in the skeleton, it calls the provided visitor function, passing the GLTF2 object,
    the node, and any additional keyword arguments to the visitor function.

    Parameters
    ----------
    gltf : GLTF2
        The GLTF2 object that contains the skeleton.
    visitor : Callable
        The visitor function to apply to each node. This function should take a GLTF2 object,
        a node, and any number of additional keyword arguments, and return None.
    root_node : Node, optional
        The node from which to start the traversal. If not provided, the traversal starts from
        the root of the skeleton.
    **kwargs
        Additional keyword arguments to pass to the visitor function. 
        Look up the specific visitor you want to use to see which arguments it accepts.
        By convention, visitor arguments that start with an _ are considered private to the visitor itself and 
        should not be used or specified by the caller.

    Returns
    -------
    None
    """
    
    # Recursive function to visit current and all child nodes
    def visit_nodes(gltf: GLTF2, curr_node_idx: int, parent_node_idx: int, **kwargs):
        # Visit the current node
        visitor(gltf, curr_node_idx, parent_node_idx, **kwargs)

        curr_node = gltf.nodes[curr_node_idx]
        
        # Visit all children of the current node recursively
        for child_joint_idx in curr_node.children:
            visit_nodes(gltf, child_joint_idx, curr_node_idx, **kwargs)

    # Start the recursive visitor with the root node
    visit_nodes(gltf, gltf.skins[0].skeleton if root_node is None else root_node, -1, **kwargs)

def skeleton_visitor_BFS(gltf: GLTF2, visitor: Callable[Concatenate[GLTF2, int, int, ...], None], root_node: Node = None, **kwargs) -> None:
    """
    Traverses the skeleton of a GLTF2 object in a breadth-first manner and applies a visitor function to each node.

    This function performs a breadth-first traversal of the skeleton, starting from the root node.
    For each node in the skeleton, it calls the provided visitor function, passing the GLTF2 object,
    the node, and any additional keyword arguments to the visitor function.

    Parameters
    ----------
    gltf : GLTF2
        The GLTF2 object that contains the skeleton.
    visitor : Callable
        The visitor function to apply to each node. This function should take a GLTF2 object,
        a node, and any number of additional keyword arguments, and return None.
    root_node : Node, optional
        The node from which to start the traversal. If not provided, the traversal starts from
        the root of the skeleton.
    **kwargs
        Additional keyword arguments to pass to the visitor function. 
        Look up the specific visitor you want to use to see which arguments it accepts.
        By convention, visitor arguments that start with an _ are considered private to the visitor itself and 
        should not be used or specified by the caller.

    Returns
    -------
    None
    """

    # Initialize a queue with the root node
    queue = deque([(gltf.skins[0].skeleton if root_node is None else root_node, -1)])

    while queue:
        # Dequeue a node and its parent
        curr_node_idx, parent_node_idx = queue.popleft()

        # Visit the current node
        visitor(gltf, curr_node_idx, parent_node_idx, **kwargs)

        # Enqueue all children of the current node
        curr_node = gltf.nodes[curr_node_idx]
        for child_joint_idx in curr_node.children:
            queue.append((child_joint_idx, curr_node_idx))

class Visitors:
    """
    Container for visitor functions that can be applied to a skeleton in a GLTF2 object
    """
    
    def plot_bindpose(gltf: GLTF2, curr_node_idx: int, parent_node_idx: int, axes: Axes, 
                      joint_color: Union[str, Tuple[float, float, float]] = 'blue', 
                      bone_color: Union[str, Tuple[float, float, float]] = 'red', 
                      _joint_transforms: Dict[str, np.ndarray] = {}):
        """
        Plots the bind pose of a GLTF model. 
        Note that any arguments starting with an _ are considered private and should not be
        used or specified by the caller

        Parameters:
        - gltf (GLTF2): The GLTF model data.
        - curr_node_idx (int): The index of the current node.
        - parent_node_idx (int): The index of the parent node.
        - ax (Axes): The matplotlib Axes object to plot on.
        - joint_color (Union[str, Tuple[float, float, float]], optional): The color of the joint markers. Defaults to 'blue'.
        - bone_color (Union[str, Tuple[float, float, float]], optional): The color of the bone lines. Defaults to 'red'.
        - _joint_transforms (Dict[str, np.ndarray], optional): Dictionary to store joint transforms. Defaults to {}.
        """
        
        # Get the current node and parent node information
        curr_node = gltf.nodes[curr_node_idx]
        parent_node_name = gltf.nodes[parent_node_idx].name if parent_node_idx != -1 else None

        # Get the parent transform if it exists, otherwise use identity
        parent_transform = _joint_transforms[parent_node_name] if parent_node_name in _joint_transforms else np.identity(4)
        # Find the parent's global position
        parent_global_pos = pt.transform(parent_transform, np.array([0, 0, 0, 1]))

        # Calculate the current joint's local transform
        curr_joint_local_transform = pt.transform_from_pq(
            np.array([curr_node.translation[0], 
                      curr_node.translation[1], 
                      curr_node.translation[2],
                      curr_node.rotation[3],
                      curr_node.rotation[0], 
                      curr_node.rotation[1], 
                      curr_node.rotation[2]]))
        # Calculate the current joint's global transform by composing the local transform and parent transform
        curr_joint_global_transform = pt.concat(curr_joint_local_transform, parent_transform)
        
        # Store the current joint's global transform for later use
        _joint_transforms[curr_node.name] = curr_joint_global_transform

        # Calculate the current joint's global position and plot the joint marker
        curr_joint_global_pos = pt.transform(curr_joint_global_transform, np.array([0, 0, 0, 1]))
        axes.scatter(curr_joint_global_pos[0], curr_joint_global_pos[1], curr_joint_global_pos[2], c = joint_color, marker='o')

        # If the parent node exists, plot the bone line connecting the parent and current joint
        if parent_node_name is not None:
            axes.plot([parent_global_pos[0], curr_joint_global_pos[0]],
                    [parent_global_pos[1], curr_joint_global_pos[1]],
                    [parent_global_pos[2], curr_joint_global_pos[2]], c = bone_color)
            
    def plot_pose_at(gltf: GLTF2, curr_node_idx: int, parent_node_idx: int,
                        time: float, animation: Dict[int, AnimationChannelData], axes: Axes, 
                        _joint_transforms: Dict[str, np.ndarray] = {}):
        """
        Visitor that plots the pose of a skeleton in a GLTF model at a specific time.

        Args:
            gltf (GLTF2): The GLTF model data.
            curr_node_idx (int): The index of the current node.
            parent_node_idx (int): The index of the parent node.
            time (float): The time at which to plot the pose.
            animation (Dict[int, AnimationChannelData]): The animation data for the model.
            ax (Axes): The matplotlib Axes object to plot on.
            _joint_transforms (Dict[str, np.ndarray], optional): Dictionary to store joint transforms. Defaults to {}.

        Returns:
            None
        """

        # Get the current joint and parent node information
        curr_joint = gltf.nodes[curr_node_idx]
        parent_node_name = gltf.nodes[parent_node_idx].name if parent_node_idx != -1 else None

        # Get the parent transform
        parent_transform = _joint_transforms[parent_node_name] if parent_node_name in _joint_transforms else np.identity(4)

        # Calculate the parent global position
        parent_global_pos = pt.transform(parent_transform, np.array([0, 0, 0, 1]))

        # Calculate the local bindpose transform of the current joint
        joint_local_transform = pt.transform_from_pq(
            np.array([curr_joint.translation[0], 
                      curr_joint.translation[1], 
                      curr_joint.translation[2],
                      curr_joint.rotation[3],
                      curr_joint.rotation[0], 
                      curr_joint.rotation[1], 
                      curr_joint.rotation[2]]))

        # Get the animation transform for the current joint. If there is no animation channel for this node, 
        # use identity as the animation transform
        joint_anim_local_transform = get_keyframe_at(time, animation[curr_node_idx]) if curr_node_idx in animation else np.identity(4)        

        # Calculate the global transform of the current joint by composing the local bindpose transform, animation transform 
        # and parent transform together
        curr_joint_global_transform = pt.concat(pt.concat(joint_local_transform, joint_anim_local_transform), parent_transform)
        
        # Store the current joint's global transform for later use
        _joint_transforms[curr_joint.name] = curr_joint_global_transform

        # Calculate the global position of the current joint
        curr_joint_global_pos = pt.transform(curr_joint_global_transform, np.array([0, 0, 0, 1]))

        # Plot the current joint's global position
        axes.scatter(curr_joint_global_pos[0], curr_joint_global_pos[1], curr_joint_global_pos[2], c='blue', marker='o')

        # If there is a parent joint, plot a line connecting the parent and current joint
        if parent_node_name is not None:
            axes.plot([parent_global_pos[0], curr_joint_global_pos[0]],
                    [parent_global_pos[1], curr_joint_global_pos[1]],
                    [parent_global_pos[2], curr_joint_global_pos[2]], c='r')

    def plot_pose_at_use_inv_bind(gltf: GLTF2, curr_node_idx: int, parent_node_idx: int,
                        time: float, animation: Dict[int, AnimationChannelData], axes: Axes, 
                        _joint_transforms: Dict[str, np.ndarray] = {},
                        _inv_bind_matrices: Dict[int, np.ndarray] = {}):
        """
        Visitor that plots the pose of a skeleton in a GLTF model at a specific time.

        Args:
            gltf (GLTF2): The GLTF model data.
            curr_node_idx (int): The index of the current node.
            parent_node_idx (int): The index of the parent node.
            time (float): The time at which to plot the pose.
            animation (Dict[int, AnimationChannelData]): The animation data for the model.
            ax (Axes): The matplotlib Axes object to plot on.
            _joint_transforms (Dict[str, np.ndarray], optional): Dictionary to store joint transforms. Defaults to {}.
            _inv_bind_matrices (Dict[int, np.ndarray]): Dictionary to store inverse bind matrices. Defaults to {}.
        Returns:
            None
        """

        # Get the current joint and parent node information
        curr_joint = gltf.nodes[curr_node_idx]
        parent_node_name = gltf.nodes[parent_node_idx].name if parent_node_idx != -1 else None

        # Get the parent transform
        parent_transform = _joint_transforms[parent_node_name] if parent_node_name in _joint_transforms else np.identity(4)
        # Calculate the parent global position
        parent_global_pos = pt.transform(parent_transform, np.array([0, 0, 0, 1]), strict_check = False)

        if len(_inv_bind_matrices) == 0:
            inverse_bind_matrices = get_accessor_data(gltf, gltf.accessors[gltf.skins[0].inverseBindMatrices])
            inverse_bind_matrices_idx = gltf.skins[0].joints
            for j in gltf.skins[0].joints:
                _inv_bind_matrices[j] = np.array(inverse_bind_matrices[inverse_bind_matrices_idx.index(j)]).reshape(4, 4).transpose()

        # Get the animation transform for the current joint. If there is no animation channel for this node, 
        # use identity as the animation transform
        joint_anim_local_transform = get_keyframe_at(time, animation[curr_node_idx]) if curr_node_idx in animation else np.identity(4)

        # Use inverse bind matrix if it exists, otherwise use identity
        # See: https://stackoverflow.com/a/41327588/802203
        joint_inv_bind_matrix = _inv_bind_matrices[curr_node_idx] if curr_node_idx in _inv_bind_matrices else np.identity(4)

        # Calculate the local transform of the current joint
        joint_local_transform = pt.concat(joint_anim_local_transform, joint_inv_bind_matrix, strict_check = False)

        # Calculate the global transform of the current joint by composing the local bindpose transform, animation transform 
        # and parent transform together
        curr_joint_global_transform = pt.concat(joint_local_transform, parent_transform, strict_check = False)
        
        # Store the current joint's global transform for later use
        _joint_transforms[curr_joint.name] = curr_joint_global_transform

        # Calculate the global position of the current joint
        curr_joint_global_pos = pt.transform(curr_joint_global_transform, np.array([0, 0, 0, 1]), strict_check = False)

        # Plot the current joint's global position
        axes.scatter(curr_joint_global_pos[0], curr_joint_global_pos[1], curr_joint_global_pos[2], c='blue', marker='o')

        # If there is a parent joint, plot a line connecting the parent and current joint
        if parent_node_name is not None:
            axes.plot([parent_global_pos[0], curr_joint_global_pos[0]],
                    [parent_global_pos[1], curr_joint_global_pos[1]],
                    [parent_global_pos[2], curr_joint_global_pos[2]], c='r')
