# To generate the virtual machine with docker and the topics_and_summary docker container execute:
# $ vagrant up
# To connect to the running virtual machine execute:
# $ vagrant ssh
# To execute the demo from inside the virtual machine execute:
# $ docker start -i topics_and_summary

Vagrant.configure("2") do |config|
  # The OS of the virtual machine will be ubuntu
  config.vm.box = "ubuntu/xenial64"

  # The virtual machine will be used with VirtualBox
  config.vm.provider "virtualbox" do |v|
    # The virtual machine will have 2GB of RAM
    v.memory = 2048
  end

  # Install docker inside the virtual machine
  config.vm.provision "docker" do |docker|
    # The argument to build an image is the path to give to docker build.
    # This must be a path that exists within the guest machine.
    # If you need to get data to the guest machine, use a synced folder.

    # The original docker build command is: "docker build . -t topics_and_summary:latest"
    # The path specified in build is the current folder, which by default is synced
    # with the /vagrant folder of the guest machine (is a shared folder).
    docker.build_image "/vagrant",
                       # Additional arguments passed to docker build
                       args: "-t topics_and_summary:latest"

    # Run the container as part of the vagrant up
    # The original docker run command is: "docker run --name topics_and_summary
    # -v $PWD/demo-images:/topics_and_summary/demo-images -i -t topics_and_summary:latest"
    docker.run "topics_and_summary",
               image: "topics_and_summary:latest",
               # Extra arguments for docker run on the command line
               # $PWD/demo-images is replaced by /vagrant/demo-images, because here we need to
               # specify paths inside the virtual machine, and the /vagrant folder is synced with
               # the project folder, that contains the Vagrantfile and the Dockerfile.
               args: "-v /vagrant/demo-images:/topics_and_summary/demo-images -i -t"
  end
end
