/**
 * This example demonstrates how to use ROS2 to send low-level motor commands of
 *unitree go2 robot
 **/
#include "common/motor_crc.h"
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/bms_cmd.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/motor_cmd.hpp"

// Create a low_level_cmd_sender class for low state receive
class LowLevelCmdSender : public rclcpp::Node {
 public:
  LowLevelCmdSender() : Node("low_level_cmd_sender") {
    // the cmd_puber is set to subscribe "/lowcmd" topic
    cmd_puber_ = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", 10);

    // The timer is set to 200Hz, and bind to
    // low_level_cmd_sender::timer_callback function
    timer_ = this->create_wall_timer(std::chrono::milliseconds(5),
                                     [this] { timer_callback(); });

    // Initialize lowcmd
    init_cmd();

    // Running time count
  }

 private:
  void timer_callback() {
    // simple sine wave motion for RL_0 joint
    double t = this->now().seconds();
    double angle = 1.0 * sin(2.0 * M_PI * 0.5 * t);  // ±0.5 rad at 0.5 Hz

    // Position control for RL_0
    cmd_msg_.motor_cmd[RL_1].q = angle;    // target angle
    cmd_msg_.motor_cmd[RL_1].kp = 60;      // higher gain
    cmd_msg_.motor_cmd[RL_1].dq = 0;       // target velocity
    cmd_msg_.motor_cmd[RL_1].kd = 1;
    cmd_msg_.motor_cmd[RL_1].tau = 0;

    // Position control for RL_1
    //cmd_msg_.motor_cmd[RL_1].q = angle;
    //cmd_msg_.motor_cmd[RL_1].kp = 60;
    //cmd_msg_.motor_cmd[RL_1].dq = 0;
    //cmd_msg_.motor_cmd[RL_1].kd = 1;
    //cmd_msg_.motor_cmd[RL_1].tau = 0;

    // keep other joints passive (no torque)
    for (int i = 0; i < 20; i++) {
        if (i != RL_1) {
            cmd_msg_.motor_cmd[i].q = PosStopF;
            cmd_msg_.motor_cmd[i].kp = 0;
            cmd_msg_.motor_cmd[i].dq = VelStopF;
            cmd_msg_.motor_cmd[i].kd = 0;
            cmd_msg_.motor_cmd[i].tau = 0;
        }
    }

    get_crc(cmd_msg_);  // check CRC
    cmd_puber_->publish(cmd_msg_);
  }

  void init_cmd() {
    for (int i = 0; i < 20; i++) {
      cmd_msg_.motor_cmd[i].mode =
          0x01;  // Set toque mode, 0x00 is passive mode
      cmd_msg_.motor_cmd[i].q = PosStopF;
      cmd_msg_.motor_cmd[i].kp = 0;
      cmd_msg_.motor_cmd[i].dq = VelStopF;
      cmd_msg_.motor_cmd[i].kd = 0;
      cmd_msg_.motor_cmd[i].tau = 0;
    }
  }

  rclcpp::TimerBase::SharedPtr timer_;  // ROS2 timer
  rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr
      cmd_puber_;  // ROS2 Publisher

  unitree_go::msg::LowCmd cmd_msg_;  // Unitree go2 lowcmd message
  double time_out_{0};               // Running time count
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);  // Initialize rclcpp
  rclcpp::TimerBase::SharedPtr const
      timer_;  // Create a timer callback object to send cmd in time intervals
  auto node =
      std::make_shared<LowLevelCmdSender>();  // Create a ROS2 node and make
                                              // share with
                                              // low_level_cmd_sender class
  rclcpp::spin(node);                         // Run ROS2 node
  rclcpp::shutdown();                         // Exit
  return 0;
}