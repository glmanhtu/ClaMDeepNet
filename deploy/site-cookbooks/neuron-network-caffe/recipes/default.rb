#
# Cookbook Name:: neuron-network-caffe
# Recipe:: default
#
# Copyright 2017, glmanhtu
#
# All rights reserved - Do Not Redistribute
#

include_recipe "poise-python"

package 'libopencv-dev'
package 'liblmdb-dev'
package 'python-dev'
package 'python-opencv'
package 'python-numpy'
package 'vim'
package 'git'

default_user = 'ubuntu'

python_runtime '2'

python_package 'lmdb'
python_package 'Mako'

source_dir = "/opt/cat-dogs"

directory source_dir do
  owner default_user
  group default_user
  mode '0755'
  action :create
end

file "/tmp/git_wrapper.sh" do
  owner default_user
  mode "0755"
  content "#!/bin/sh\nexec /usr/bin/ssh -o \"StrictHostKeyChecking=no\""
end

git "#{source_dir}/repo" do
  repository "https://github.com/glmanhtu/deeplearning-pagoda.git"
  branch "master"
  action :sync
  group default_user
  user default_user
  ssh_wrapper "/tmp/git_wrapper.sh"
end