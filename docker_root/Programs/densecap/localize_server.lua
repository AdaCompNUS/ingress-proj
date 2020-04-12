local ros = require 'ros'
require 'ros.actionlib.ActionServer'
local actionlib = ros.actionlib

local image = require 'image'
gm = require 'graphicsmagick'

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
require 'camera'
require 'qt'
require 'qttorch'
require 'qtwidget'

require 'densecap.DenseCapModel'
require 'densecap.modules.BoxIoU'

local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'
local vis_utils = require 'densecap.vis_utils'

cmd = torch.CmdLine()
cmd:option('-checkpoint',
  'data/models/densecap/densecap-pretrained-vgg16.t7')
cmd:option('-display_image_height', 640)
cmd:option('-display_image_width', 480)
cmd:option('-model_image_size', 720)
cmd:option('-num_proposals', 500)
cmd:option('-boxes_to_show', 40)
cmd:option('-webcam_fps', 1)
cmd:option('-gpu', 0)
cmd:option('-timing', 1)
cmd:option('-detailed_timing', 0)
cmd:option('-text_size', 2)
cmd:option('-box_width', 2)
cmd:option('-rpn_nms_thresh', 0.7)
-- cmd:option('-final_nms_thresh', 0.3)
cmd:option('-final_nms_thresh', 0.05)

cmd:option('-use_cudnn', 1)

ros.init('localize_actionserver')
nh = ros.NodeHandle()

spinner = ros.AsyncSpinner()
spinner:start()

local function grab_frame(opt, img_orig)

  -- local img_orig = img
  -- local img = image.scale(img_orig, opt.model_image_size)
  local img = img_orig
  local img_caffe = img:index(1, torch.LongTensor{3, 2, 1}):mul(255)
  local vgg_mean = torch.Tensor{103.939, 116.779, 123.68}
  img_caffe:add(-1, vgg_mean:view(3, 1, 1):expandAs(img_caffe))
  local H, W = img_caffe:size(2), img_caffe:size(3)
  img_caffe = img_caffe:view(1, 3, H, W)

  return img_orig, img_caffe
end
  
local function Localize_Action_Server(goal_handle)
  ros.INFO("Localize_Action_Server")
  local g = goal_handle:getGoal().goal

  print (g)

  -- Convert to torch image tensor
  local img_tensor = torch.reshape(g.input.data, torch.LongStorage{g.input.height, g.input.width, 3})
  local img_gm = gm.Image(img_tensor, 'RGB', 'DWH')
  -- local img_gm = gm.Image(img_tensor, 'RGB', 'DWH')
  local img = img_gm:toTensor('double','RGB', 'DHW')

  goal_handle:setAccepted('yip')
  
  -- compute boxes
  local img_orig, img_caffe = grab_frame(opt, img)
  local boxes_xcycwh, scores, captions, feats = model:forward_test(img_caffe:type(dtype))

  -- compute fc7 features for whole image
  local whole_img_roi = torch.FloatTensor{{1.0, 1.0, g.input.width*1.0, g.input.height*1.0}}
  local out = model:forward_boxes(img_caffe:type(dtype), whole_img_roi:type(dtype))
  local f_objectness_scores, f_seqs, f_roi_codes, f_hidden_codes, f_captions = unpack(out)

  -- scale boxes to image size
  boxes_xywh = box_utils.xcycwh_to_xywh(boxes_xcycwh)
  local scale = img_orig:size(2) / img_caffe:size(3)
  boxes_xywh = box_utils.scale_boxes_xywh(boxes_xywh, scale)

  if clear_history_every_localize then
    history_feats = {}
    history_captions = {}
    history_boxes_xcycwh = {}
    history_boxes_xywh = {}
  end

  -- store results for future queries
  history_feats[g.frame_id] = feats:type('torch.FloatTensor')
  history_captions[g.frame_id] = captions
  history_boxes_xcycwh[g.frame_id] = boxes_xcycwh:type('torch.FloatTensor')
  history_boxes_xywh[g.frame_id] = boxes_xywh:type('torch.FloatTensor')

  -- return results
  local r = goal_handle:createResult()
  r.fc7_img = f_roi_codes:reshape(f_roi_codes:size(2)):float()
  r.fc7_vecs = feats:reshape(feats:size(1) * feats:size(2)):float()
  r.boxes = boxes_xywh:reshape(boxes_xywh:size(1) * boxes_xywh:size(2)):float()
  r.scores = scores:reshape(scores:size(1)):float()
  r.captions = captions

  goal_handle:setSucceeded(r, 'done')
end


local function Localize_Probs_Action_Server(goal_handle)
  ros.INFO("Localize_Probs_Action_Server")
  local g = goal_handle:getGoal().goal

  print (g)

  -- Convert to torch image tensor
  local img_tensor = torch.reshape(g.input.data, torch.LongStorage{g.input.height, g.input.width, 3})
  local img_gm = gm.Image(img_tensor, 'RGB', 'DWH')
  -- local img_gm = gm.Image(img_tensor, 'RGB', 'DWH')
  local img = img_gm:toTensor('double','RGB', 'DHW')

  goal_handle:setAccepted('yip')
  
  -- compute boxes
  local img_orig, img_caffe = grab_frame(opt, img)
  local boxes_xcycwh, scores, captions, feats, word_probs = model:forward_test(img_caffe:type(dtype))
  local probs_copy = torch.Tensor(word_probs:size()):copy(word_probs)
  -- print (probs_copy:size())

  -- compute fc7 features for whole image
  local whole_img_roi = torch.FloatTensor{{1.0, 1.0, g.input.width*1.0, g.input.height*1.0}}
  local out = model:forward_boxes(img_caffe:type(dtype), whole_img_roi:type(dtype))
  local f_objectness_scores, f_seqs, f_roi_codes, f_hidden_codes, f_captions = unpack(out)

  -- scale boxes to image size
  boxes_xywh = box_utils.xcycwh_to_xywh(boxes_xcycwh)
  local scale = img_orig:size(2) / img_caffe:size(3)
  boxes_xywh = box_utils.scale_boxes_xywh(boxes_xywh, scale)

  if clear_history_every_localize then
    history_feats = {}
    history_captions = {}
    history_boxes_xcycwh = {}
    history_boxes_xywh = {}
  end

  -- store results for future queries
  history_feats[g.frame_id] = feats:type('torch.FloatTensor')
  history_captions[g.frame_id] = captions
  history_boxes_xcycwh[g.frame_id] = boxes_xcycwh:type('torch.FloatTensor')
  history_boxes_xywh[g.frame_id] = boxes_xywh:type('torch.FloatTensor')

  -- return results
  local r = goal_handle:createResult()
  r.fc7_img = f_roi_codes:reshape(f_roi_codes:size(2)):float()
  r.fc7_vecs = feats:reshape(feats:size(1) * feats:size(2)):float()
  r.word_probs = probs_copy:reshape(probs_copy:size(1) * probs_copy:size(2) * probs_copy:size(3)):float()
  r.boxes = boxes_xywh:reshape(boxes_xywh:size(1) * boxes_xywh:size(2)):float()
  r.scores = scores:reshape(scores:size(1)):float()
  r.captions = captions

  goal_handle:setSucceeded(r, 'done')
end


local function Query_Action_Goal(goal_handle)
  ros.INFO("Query_Action_Goal")
  local g = goal_handle:getGoal().goal

  print (g)

  -- TODO IMPORTANT: check if history is available, otherwise reject the goal
  goal_handle:setAccepted('yip')

  -- local top_k_ids, top_k_boxes, top_k_losses, top_k_meteor_ranks, search_time = search(g.query, g.min_loss_threshold)
  local top_k_ids, top_k_boxes, top_k_losses, top_k_meteor_ranks, search_time, top_k_feats, top_k_orig_idx, top_k_meteor_scores = model:language_query(history_feats, history_captions, history_boxes_xcycwh, history_boxes_xywh, g.query, g.min_loss_threshold, g.k)

  local r = goal_handle:createResult()
  r.frame_ids = top_k_ids:reshape(top_k_ids:size(1)):int()
  r.captioning_losses = top_k_losses:reshape(top_k_losses:size(1)):float()
  r.boxes = top_k_boxes:reshape(top_k_boxes:size(1) * top_k_boxes:size(2)):float()
  r.meteor_ranks = top_k_meteor_ranks:reshape(top_k_meteor_ranks:size(1)):int()
  r.search_time = search_time
  r.fc7_vecs = top_k_feats:reshape(top_k_feats:size(1) * top_k_feats:size(2)):float()
  r.orig_idx = top_k_orig_idx:reshape(top_k_orig_idx:size(1)):int()
  r.meteor_scores = top_k_meteor_scores:reshape(top_k_meteor_scores:size(1)):float()

  goal_handle:setSucceeded(r, 'done')
end

local function Extract_Action_Goal(goal_handle)
  ros.INFO("Extract_Action_Goal")
  local g = goal_handle:getGoal().goal

  print (g)

  -- Convert to torch image tensor
  local img_tensor = torch.reshape(g.input.data, torch.LongStorage{g.input.height, g.input.width, 3})
  local img_gm = gm.Image(img_tensor, 'RGB', 'DWH')
  -- local img_gm = gm.Image(img_tensor, 'RGB', 'DWH')
  local img = img_gm:toTensor('double','RGB', 'DHW')
  local img_orig, img_caffe = grab_frame(opt, img)

  goal_handle:setAccepted('yip')

  -- compute fc7 features for boxes
  local boxes_xcycwh = torch.reshape(g.boxes, g.boxes:size(1)/4, 4)
  local out = model:forward_boxes(img_caffe:type(dtype), boxes_xcycwh:type(dtype))
  local f_objectness_scores, f_seqs, f_roi_codes, f_hidden_codes, f_captions = unpack(out)

  -- scale boxes to image size
  boxes_xywh = box_utils.xcycwh_to_xywh(boxes_xcycwh)
  local scale = img_orig:size(2) / img_caffe:size(3)
  boxes_xywh = box_utils.scale_boxes_xywh(boxes_xywh, scale)

  if clear_history_every_localize then
    history_feats = {}
    history_captions = {}
    history_boxes_xcycwh = {}
    history_boxes_xywh = {}
  end

  -- store results for future queries
  history_feats[g.frame_id] = f_roi_codes:type('torch.FloatTensor')
  history_captions[g.frame_id] = f_captions
  history_boxes_xcycwh[g.frame_id] = boxes_xcycwh:type('torch.FloatTensor')
  history_boxes_xywh[g.frame_id] = boxes_xywh:type('torch.FloatTensor')

  -- return results
  local r = goal_handle:createResult()
  r.fc7_vecs = f_roi_codes:reshape(f_roi_codes:size(1) * f_roi_codes:size(2)):float()
  r.scores = f_objectness_scores:reshape(f_objectness_scores:size(1)):float()
  r.captions = f_captions

  goal_handle:setSucceeded(r, 'done')
end



opt = cmd:parse(arg)

-- Setup Localization Server
local as_localize_server = actionlib.ActionServer(nh, 'dense_localize', 'ingress_msgs/Localize')
local as_localize_probs_server = actionlib.ActionServer(nh, 'dense_localize_probs', 'ingress_msgs/LocalizeProbs')
local as_query_server = actionlib.ActionServer(nh, 'localize_query', 'ingress_msgs/LocalizeQuery')
local as_extract_server = actionlib.ActionServer(nh, 'extract_features', 'ingress_msgs/ExtractFeatures')

as_localize_server:registerGoalCallback(Localize_Action_Server)
as_localize_probs_server:registerGoalCallback(Localize_Probs_Action_Server)
as_query_server:registerGoalCallback(Query_Action_Goal)
as_extract_server:registerGoalCallback(Extract_Action_Goal)

print('Starting Dense Localization and Query action server...')
as_localize_server:start()
as_localize_probs_server:start()
as_query_server:start()
as_extract_server:start()

opt = cmd:parse(arg)
dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn)
-- dtype, use_cudnn = utils.setup_gpus(-1, 0)


-- Load the checkpoint
print('loading checkpoint from ' .. opt.checkpoint)
checkpoint = torch.load(opt.checkpoint)
model = checkpoint.model
print('done loading checkpoint')

-- Ship checkpoint to GPU and convert to cuDNN
model:convert(dtype, use_cudnn)
model:setTestArgs{
  rpn_nms_thresh = opt.rpn_nms_thresh,
  final_nms_thresh = opt.final_nms_thresh,
  num_proposals = opt.num_proposals,
}
model:evaluate()

-- NOTE: linear space complexity
clear_history_every_localize = true

history_feats = {}
history_captions = {}
history_boxes_xcycwh = {}
history_boxes_xywh = {}

timer = torch.Timer()

local s = ros.Duration(0.001)
while ros.ok() do
  s:sleep()
  ros.spinOnce()
end

as_localize_server:shutdown()
as_localize_probs_server:shutdown()
as_query_server:shutdown()
as_extract_server:shutdown()
nh:shutdown()
server:shutdown()
ros.shutdown()
