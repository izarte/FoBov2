from setuptools import setup

setup(
    name="fobo2_env",
    version="0.0.1",
    install_requires=["gymnasium==0.29.1", "pybullet==3.2.6", "pyb-utils==2.2.0"],
    package_data={
        "fobo2_env": ["src/models/*"],
    },
)
