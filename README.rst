################################
pl-mrinet-s2i
################################


Abstract
********

| Sample Tensorflow application plugin for ChRIS Project.
| The application is a digit-identification application based on MNIST data.
| The intent is to make it easy for future users to take this sample and use it as the starting point to build your own plugin.

Build
*****

.. note::
  Make sure you have 's2i' installed where you are building.

On a Fedora system the command would be:

.. code-block:: bash

  sudo dnf install s2i

Build The Base Container:
=========================

From the root directory of the git repository, run the following command to build the container:

.. code-block:: bash

    make build

The default image built will be named:

``pl-mrinet-s2i-centos-python3``

This default container image is a CentOS image that includes Python3.

.. note::
  See 'Makefile' if you'd like to use a different IMAGE_NAME



S2I Build & Training
====================

Run the s2i command below to train the model and build the sample application plugin

``s2i build <source-location> <builder-image-name> <output-image-name>``

Example Command: (Run from root of project repo)

.. code-block:: bash

  s2i build . pl-mrinet-s2i-centos-python3 mrinet-s2i-centos

If you'd like see additional information when building, append the --loglevel <loglevel_value>

.. code-block:: bash

  s2i build . pl-mrinet-s2i-centos-python3 mrinet-s2i-centos --loglevel 5


The output of the above command is a container named:
``mrinet-s2i-centos``

Run
*****
Using Docker run
====================

Start Docker Service

.. code-block:: bash

   sudo systemctl start docker

Start Docker daemon at boot (Optional)

.. code-block:: bash

   sudo systemctl enable docker

Make sure your user is in the docker group if you want to run the docker command as a non-root user

.. code-block:: bash

   sudo groupadd docker && sudo gpasswd -a ${USER} docker && sudo systemctl restart docker
   newgrp docker


Run training or inference
==========================

For the first time you must run training mode first and then run inference

.. code-block:: bash

 docker run mrinet-s2i-centos ./tensorflowapp-training.py  --run_mode train /opt/app-root/src/input /opt/app-root/src/output




