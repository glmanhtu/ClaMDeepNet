name "default"
description "Install Neuron Network Caffe"

run_list(
  "recipe[opencv]",
  "recipe[neuron-network-caffe]"
)