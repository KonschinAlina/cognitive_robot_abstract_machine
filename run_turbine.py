import rclpy
import time
import math
from pathlib import Path
from rclpy.node import Node
from sensor_msgs.msg import JointState

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world import World
from semantic_digital_twin.semantic_annotations.semantic_annotations import MainShaft


class RVizBridgeNode(Node):
    def __init__(self):
        super().__init__("turbine_rviz_bridge")
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)

    def publish_world_joints(self, world: World):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        names = []
        positions = []

        # read joints from world
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


def main():
    # init ros node
    rclpy.init()
    rviz_node = RVizBridgeNode()

    # Path to urdf file
    base_dir = Path(__file__).parent
    urdf_path = base_dir / "semantic_digital_twin" / "resources" / "urdf" / "windturbine.urdf"
    
    # Parse urdf to world
    world = URDFParser.from_file(str(urdf_path2), "windturbine.urdf").parse()
    #print(world.bodies)

    # Get body 'MainShaft' from world
    # bisher: from name instead of type
    bodies = world.get_kinematic_structure_entities_by_name("main_shaft")
    main_shaft_body = bodies[0] if isinstance(bodies, list) else bodies

    ### add semantic annotation for body
    annotation = MainShaft(body=main_shaft_body)
    with world.modify_world():
        world.add_semantic_annotation(annotation)

    #
    angle = 0.0
    target_fps = 30.0  # 30 frames/sec
    frame_duration = 1.0 / target_fps
    speed_rad_per_sec = 0.5 # speed (rad/s)
    last_time = time.time()

    try:
        while rclpy.ok():
            now = time.time()
            dt = now - last_time
            last_time = now

            # 1. calculate angle
            angle = (angle + speed_rad_per_sec * dt) % (2 * math.pi)
            #angle = (angle + speed_rad_per_sec) % (2 * math.pi)

            # 2. Set position in world model
            annotation.rotate(angle)

            # 3. publish joints
            rviz_node.publish_world_joints(world)

            # 4. process ROS-Queue
            rclpy.spin_once(rviz_node, timeout_sec=0.0)

            # 5. frame rate
            #rate.sleep()
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
