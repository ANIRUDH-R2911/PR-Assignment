from setuptools import find_packages, setup

package_name = 'object_detection_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anirudh',
    maintainer_email='anirudh@todo.todo',
    description='TODO: Package description',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection = object_detection_segmentation.object_detection:main',
            'semantic_segmentation = object_detection_segmentation.semantic_segmentation:main',
        ],
    },
)
