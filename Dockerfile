FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:1
ENV ROS_DISTRO=noetic
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV QT_X11_NO_MITSHM=1

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg2 && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list

RUN apt-get update -o Acquire::Retries=5 && apt-get install -y --no-install-recommends \
    git curl wget \
    xfce4 xfce4-goodies dbus-x11 \
    xvfb x11vnc \
    supervisor \
    novnc websockify \
    python3-pip python3-rosdep \
    ros-noetic-moveit \
    ros-noetic-ur5-moveit-config \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-ur-description \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    ros-noetic-image-transport \
    ros-noetic-image-view \
    ros-noetic-cv-bridge \
    python3-opencv \
    python3-numpy \
    ros-noetic-gazebo-msgs \
    ros-noetic-xacro \
    ros-noetic-controller-manager \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    && rm -rf /var/lib/apt/lists/*

RUN rm -rf /opt/novnc && \
    git clone https://github.com/novnc/noVNC.git /opt/novnc && \
    git clone https://github.com/novnc/websockify.git /opt/novnc/utils/websockify && \
    cd /opt/novnc/utils/websockify && pip3 install .

RUN rosdep init || true

RUN mkdir -p /root/.gazebo/worlds /saved_worlds

WORKDIR /workspace

RUN mkdir -p /workspace/catkin_ws/src

COPY ros_ws/src/smart_arm_demo /workspace/catkin_ws/src/smart_arm_demo

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    rosdep update && \
    rosdep install --from-paths /workspace/catkin_ws/src --ignore-src -r -y || true && \
    cd /workspace/catkin_ws && catkin_make"

RUN echo 'source /opt/ros/noetic/setup.bash' >> /root/.bashrc && \
    echo 'source /workspace/catkin_ws/devel/setup.bash' >> /root/.bashrc

RUN mkdir -p /etc/supervisor/conf.d
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY start.sh /start.sh
RUN sed -i 's/\r$//' /start.sh && \
    sed -i 's/\r$//' /workspace/catkin_ws/src/smart_arm_demo/scripts/*.py && \
    chmod +x /start.sh && \
    chmod +x /workspace/catkin_ws/src/smart_arm_demo/scripts/*.py

EXPOSE 6080 5900 11311
CMD ["/start.sh"]

