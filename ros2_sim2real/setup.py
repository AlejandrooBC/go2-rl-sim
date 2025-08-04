from setuptools import setup
import os
from glob import glob

package_name = "go2_policy_deploy"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "walking_policy"),
            glob("go2_policy_deploy/walking_policy/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Alejandro Begara Criado",
    maintainer_email="aleebcschool@gmail.com",
    description="Go2 walking policy deployment node",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sim2real_node = go2_policy_deploy.sim2real_node:main"
        ],
    },
)