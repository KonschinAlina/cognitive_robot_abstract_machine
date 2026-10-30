import rclpy
import time
import math
from pathlib import Path
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world import World
from semantic_digital_twin.semantic_annotations.semantic_annotations import MainShaft, Blade


class RVizBridgeNode(Node):
    def __init__(self):
        super().__init__("turbine_rviz_bridge")
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

    def publish_world_joints(self, world: World):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        names = []
        positions = []

        # read joints from world that are not fixed
        # check for dof
        for connection in world.connections:
           if hasattr(connection, "dof") and connection.dof is not None:
                names.append(
                    connection.name.name
                    if hasattr(connection, "name")
                    else connection.child.name.name
                )
                positions.append(float(connection.position))

        if names:
            msg.name = names
            msg.position = positions
            self.publisher.publish(msg)

        # Send every connection to /tf to reset tree in RViz2
        for connection in world.connections:
            if hasattr(connection, "parent") and hasattr(connection, "child"):
                self._send_tf(connection)

    def _send_tf(self, connection):
        try:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()

            p_name = (
                connection.parent.name.name
                if hasattr(connection.parent.name, "name")
                else str(connection.parent.name)
            )

            c_name = (
                connection.child.name.name
                if hasattr(connection.child.name, "name")
                else str(connection.child.name)
            )

            t.header.frame_id = p_name
            t.child_frame_id = c_name

            matrix = (
                    getattr(connection, "parent_T_child_connection", None)
                    or getattr(connection, "parent_T_child", None)
                    or getattr(connection, "origin", None)
                    or getattr(connection, "parent_T_parent_connection", None)
            )

            if matrix is not None:
                t.transform.translation.x = float(matrix[0, 3])
                t.transform.translation.y = float(matrix[1, 3])
                t.transform.translation.z = float(matrix[2, 3])

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

            self.tf_broadcaster.sendTransform(t)

        except Exception as e:
            self.get_logger().error(f"TF Error: Connection '{connection}': {e}")

def main():
    # init ros node
    rclpy.init()
    rviz_node = RVizBridgeNode()

    # Path to urdf file
    base_dir = Path(__file__).parent
    urdf_path = base_dir / "semantic_digital_twin" / "resources" / "urdf" / "windturbine.urdf"
    
    # Parse urdf to world
    world = URDFParser.from_file(str(urdf_path), "windturbine.urdf").parse()
    #print(world.bodies)

    # reset joint positions
    # if assembling was executed first
    for conn in world.connections:
        if hasattr(conn, "position"):
            conn.position = 0.0
    world.update_forward_kinematics()

    # Get body 'MainShaft' from world
    # bisher: from name instead of type
    bodies = world.get_kinematic_structure_entities_by_name("main_shaft")
    main_shaft_body = bodies[0] if isinstance(bodies, list) else bodies

    # get blade bodies
    blade_names = ["blade_1", "blade_2", "blade_3"]
    blade_bodies = []
    #blade_bodies = world.get_kinematic_structure_entities_by_name("blade")

    for name in blade_names:
        found = world.get_kinematic_structure_entities_by_name(name)
        if found:
            body = found[0] if isinstance(found, list) else found
            blade_bodies.append(body)
            #print(blade_bodies)

    ### add semantic annotation for body
    annotation = MainShaft(body=main_shaft_body)
    with world.modify_world():
        world.add_semantic_annotation(annotation)

    blades = []
    with world.modify_world():
        for blade_body in blade_bodies:
            blade_annotation = Blade(body=blade_body)
            world.add_semantic_annotation(blade_annotation)
            blades.append(blade_annotation)

    #print(blades)

    # set blades to X rad (i.e. 0.26)
    #for blade in blades:
    #    blade.rotate(0.26)
    #    print("rotated blade")


    angle = 0.0
    target_fps = 30.0  # 30 frames/sec
    frame_duration = 1.0 / target_fps
    speed_rad_per_sec = 0.5 # speed (rad/s)
    last_time = time.time()

    init_end = time.time() + 1.0
    while time.time() < init_end and rclpy.ok():
        rviz_node.publish_world_joints(world)
        rclpy.spin_once(rviz_node, timeout_sec=0.02)
        time.sleep(0.02)

    try:
        while rclpy.ok():
            now = time.time()
            dt = now - last_time
            last_time = now

            # 1. calculate angle
            angle = (angle + speed_rad_per_sec * dt) % (2 * math.pi)
            #angle = (angle + speed_rad_per_sec) % (2 * math.pi)

            pitch_angle_blade = (math.sin(now * 1.5) + 1.0) * 0.39

            # 2. Set position in world model
            annotation.rotate(angle)

            for blade in blades:
                blade.rotate(pitch_angle_blade)

            # 3. publish joints
            rviz_node.publish_world_joints(world)

            # 4. process ROS-Queue
            rclpy.spin_once(rviz_node, timeout_sec=0.0)

            # 5. frame rate
            elapsed = time.time() - now
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            #-----------------
            # calls rotate
            #angle += 0.01
            #annotation.rotate(angle)

            # Gelenkzustände an ROS 2 / RViz senden
            #rviz_node.publish_world_joints(world)

            # ROS-Callbacks verarbeiten
            #rclpy.spin_once(rviz_node, timeout_sec=0.01)
            #time.sleep(0.02)

    except KeyboardInterrupt:
        print("Simulation ended.")
    finally:
        rviz_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
