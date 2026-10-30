import numpy as np
import math
from pathlib import Path
import time
from rclpy.node import Node
import rclpy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world import World, FixedConnection

from semantic_digital_twin.semantic_annotations.assembly_state import AssemblyState


class RVizBridgeNode(Node):
    def __init__(self):
        super().__init__("tower_assembly_bridge")
        self.publisher = self.create_publisher(JointState, "/joint_states", 1)
        self.tf_broadcaster = TransformBroadcaster(self)

    def publish_world_joints(self, world: World):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        names = []
        positions = []

        # get all connections from world
        for connection in world.connections:
            if hasattr(connection, "dof") and connection.dof is not None:
                joint_name = (
                    connection.name.name
                    if hasattr(connection, "name") and hasattr(connection.name, "name")
                    else connection.child.name.name
                )
                names.append(joint_name)
                positions.append(float(connection.position))

        if names:
            msg.name = names
            msg.position = positions
            self.publisher.publish(msg)

        # send every connection to /tf
        for connection in world.connections:
            if hasattr(connection, "parent") and hasattr(connection, "child"):
                self._send_tf(connection)

    def _send_tf(self, connection):
        try:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()

            # parent name
            p_name = (
                connection.parent.name.name
                if hasattr(connection.parent.name, "name")
                else str(connection.parent.name)
            )

            # child name
            c_name = (
                connection.child.name.name
                if hasattr(connection.child.name, "name")
                else str(connection.child.name)
            )

            t.header.frame_id = p_name
            t.child_frame_id = c_name

            # create matrix
            matrix = getattr(connection, "origin", None) or getattr(connection, "parent_T_parent_connection", None)

            # transform matrix
            if matrix is not None:
                t.transform.translation.x = float(matrix[0, 3])
                t.transform.translation.y = float(matrix[1, 3])
                t.transform.translation.z = float(matrix[2, 3])

                # convert rotation matrix to quaternions
                r00, r01, r02 = float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])
                r10, r11, r12 = float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])
                r20, r21, r22 = float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])

                trace = r00 + r11 + r22
                if trace > 0:
                    s = 0.5 / math.sqrt(trace + 1.0)
                    qw = 0.25 / s
                    qx = (r21 - r12) * s
                    qy = (r02 - r20) * s
                    qz = (r10 - r01) * s
                elif (r00 > r11) and (r00 > r22):
                    s = 2.0 * math.sqrt(1.0 + r00 - r11 - r22)
                    qw = (r21 - r12) / s
                    qx = 0.25 * s
                    qy = (r01 + r10) / s
                    qz = (r02 + r20) / s
                elif r11 > r22:
                    s = 2.0 * math.sqrt(1.0 + r11 - r00 - r22)
                    qw = (r02 - r20) / s
                    qx = (r01 + r10) / s
                    qy = 0.25 * s
                    qz = (r12 + r21) / s
                else:
                    s = 2.0 * math.sqrt(1.0 + r22 - r00 - r11)
                    qw = (r10 - r01) / s
                    qx = (r02 + r20) / s
                    qy = (r12 + r21) / s
                    qz = 0.25 * s

                t.transform.rotation.x = qx
                t.transform.rotation.y = qy
                t.transform.rotation.z = qz
                t.transform.rotation.w = qw
            else:
                t.transform.rotation.w = 1.0

            # send transform
            self.tf_broadcaster.sendTransform(t)

        except Exception as e:
            self.get_logger().error(f"TF Error: Connection '{connection}': {e}")



class WindTurbineAssembler:
    def __init__(self, world, build_order, spacing=15.0, ground_z=0.0):
        self.world = world
        self.build_order = build_order # custom build order, no rules for build order yet
        self.spacing = spacing
        self.ground_z = ground_z
        self.final_poses = {}        # name -> world_T_body (assembled)
        self.next_step = 0

        self.names = []
        for parent, child in build_order:
            for n in (parent, child):
                if n not in self.names:
                    self.names.append(n)

    # get body (by name)
    def _get_body(self, name):
        found = self.world.get_kinematic_structure_entities_by_name(name)
        return found[0] if isinstance(found, list) else found

    # get root
    def _root(self):
        # root = body without parent connection
        b = self._get_body(self.names[0])
        while getattr(b, "parent_connection", None) is not None:
            b = b.parent_connection.parent
        return b


    def _world_T_body(self, body):
        # chain of connection origin multiplied bis zur root
        T = None
        b = body
        while getattr(b, "parent_connection", None) is not None:
            conn = b.parent_connection
            T = conn.origin if T is None else conn.origin @ T
            b = conn.parent

        if T is not None:
            return T
        else:
            return HomogeneousTransformationMatrix()

    # update assembly state attribute on semantic annotations
    def _update_annotation_state(self, part_name: str, state: AssemblyState):
        for annotation in self.world.semantic_annotations:
            if hasattr(annotation, "body") and annotation.body is not None:
                body_name = (
                    annotation.body.name.name
                    if hasattr(annotation.body.name, "name")
                    else str(annotation.body.name)
                )
                if body_name == part_name and hasattr(annotation, "assembly_state"):
                    annotation.assembly_state = state
            elif hasattr(annotation, "assembly_state"):
                # Fallback, falls der Name direkt am Annotation-Objekt hängt
                annot_name = (
                    annotation.name.name
                    if hasattr(annotation, "name") and hasattr(annotation.name, "name")
                    else str(getattr(annotation, "name", ""))
                )
                if part_name in annot_name:
                    annotation.assembly_state = state

    # here the reassembling takes place, delete old connections, add new
    def _reconnect(self, parent_body, child_body, parent_T_child):
        old_conn = getattr(child_body, "parent_connection", None)
        new_conn = FixedConnection(
            parent=parent_body,
            child=child_body,
            parent_T_connection_expression=parent_T_child,
        )
        with self.world.modify_world():
            if old_conn:
                self.world.remove_connection(old_conn)
            self.world.add_connection(new_conn)
        self.world.update_forward_kinematics()

    # initial state, where parts just lay on the ground
    def lay_out_on_ground(self):
        # target pose merken till every part is assembled
        for name in self.names:
            self.final_poses[name] = self._world_T_body(self._get_body(name))
            #print(f"target poses:{self.final_poses[name]}")

        root = self._root()
        cols = 4

        for i, name in enumerate(self.names):
            row = i // cols
            col = i % cols

            x_pos = col * self.spacing
            y_pos = row * self.spacing

            pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=x_pos, y=y_pos, z=self.ground_z)

            self._reconnect(root, self._get_body(name), pose)
            self._update_annotation_state(name, AssemblyState.UNASSEMBLED)

        self.next_step = 0

    def assemble_next(self):
        if self.next_step >= len(self.build_order):
            return False
        parent, child = self.build_order[self.next_step]
        rel = self.final_poses[parent].inverse() @ self.final_poses[child]
        self._reconnect(self._get_body(parent), self._get_body(child), rel)

        # Update semantic annotation state to ASSEMBLED
        self._update_annotation_state(child, AssemblyState.ASSEMBLED)

        print(f"Step {self.next_step + 1}/{len(self.build_order)}: connected {child} to {parent}")
        self.next_step += 1
        return True

def main():
    rclpy.init()
    rviz_node = RVizBridgeNode()

    base_dir = Path(__file__).parent
    urdf_path = base_dir / "semantic_digital_twin" / "resources" / "urdf" / "windturbine.urdf"
    world = URDFParser.from_file(str(urdf_path), "windturbine.urdf").parse()
    world.update_forward_kinematics()

    order = [
        ("tower_section_1", "tower_section_2"),
        ("tower_section_2", "tower_section_3"),
        ("tower_section_3", "tower_section_4"),
        ("tower_section_4", "nacelle"),
        ("nacelle", "hub"),
        ("hub", "blade_1"),
        ("hub", "blade_2"),
        ("hub", "blade_3"),
    ]

    assembler = WindTurbineAssembler(world, order, spacing=12.0)
    print("lay parts out on ground")
    assembler.lay_out_on_ground()
    print("lay out on ground - finished")

    target_fps = 30.0
    frame_duration = 1.0 / target_fps
    step_interval = 3.0
    next_time = time.time() + step_interval

    try:
        while rclpy.ok():
            now = time.time()

            if now >= next_time:
                if assembler.assemble_next():
                    next_time = now + step_interval
                else:
                    print("\nFinished assembling.")
                    # publish final state again for rviz
                    rviz_node.publish_world_joints(world)
                    break

            rviz_node.publish_world_joints(world)
            rclpy.spin_once(rviz_node, timeout_sec=0.0)

            sleep_time = frame_duration - (time.time() - now)
            if sleep_time > 0:
                time.sleep(sleep_time)


    except KeyboardInterrupt:
        print("\nSimulation ended.")
    finally:
        rviz_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()