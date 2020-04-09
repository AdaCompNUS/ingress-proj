# Action Controller
Custom messages for dense refexp

# Installation

Clone and Make:
```bash
$ cd <your_catkin_ws>/src
$ git clone https://github.com/MohitShridhar/action_controller.git

$ cd <your_catkin_ws>
$ catkin_make
```

# Usage

Make Peacock server your rosmaster in your `~/.bashrc`:
```
export ROS_IP=<your_ip_address>
export ROS_MASTER_URI=http://172.26.186.217:11311
export ROS_HOSTNAME=<your_ip_address>

```
export your bashrc.


Check if query server is available:
```bash
$ rostopic list 
```
You should see these topic:
```
/dense_localize/cancel
/dense_localize/feedback
/dense_localize/goal
/dense_localize/result
/dense_localize/status
/dense_refexp_load/cancel
/dense_refexp_load/feedback
/dense_refexp_load/goal
/dense_refexp_load/result
/dense_refexp_load/status
/dense_refexp_load_query/cancel
/dense_refexp_load_query/feedback
/dense_refexp_load_query/goal
/dense_refexp_load_query/result
/dense_refexp_load_query/status
/dense_refexp_query/cancel
/dense_refexp_query/feedback
/dense_refexp_query/goal
/dense_refexp_query/result
/dense_refexp_query/status
```

Run Example:
```
cd <your_catkin_ws>/src/action_controller/test
python dense_refexp_example.py
```

search for 'the rightmost red cup'

See `dense_query_example.py` for usage


