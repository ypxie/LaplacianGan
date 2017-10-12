-- Modification from the codebase of scott's icml16
-- please check https://github.com/reedscot/icml2016 for details

require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'lfs'
require 'torch'



torch.setdefaulttensortype('torch.FloatTensor')

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i
end
ivocab = {}
for k,v in pairs(dict) do
  ivocab[v] = k
end

opt = {
  filenames = 'Data/birds/example_captions.t7',
  doc_length = 201,
  queries = 'Data/birds/example_captions.txt',
  net_txt = 'demo/text_encoder/lm_sje_nc4_cub_hybrid_gru18_a1_c512_0.00070_1_10_trainvalids.txt_iter30000.t7',
  path = 'Data/coco/val2014_ex_t7/'
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

net_txt = torch.load(opt.net_txt)
if net_txt.protos ~=nil then net_txt = net_txt.protos.enc_doc end


net_txt:evaluate()

-- Extract all text features.
local fea_txt = {}
-- Decode text for sanity check.
local raw_txt = {}
local raw_img = {}
for query_str in io.lines(opt.queries) do
  print(query_str)
  local txt = torch.zeros(1, opt.doc_length, #alphabet)
  for t = 1,opt.doc_length do
    local ch = query_str:sub(t,t)
    local ix = dict[ch]
    if ix ~= 0 and ix ~= nil then
      txt[{1,t,ix}] = 1
    end
  end
  raw_txt[#raw_txt+1] = query_str
  txt = txt:cuda()
  feat = net_txt:forward(txt):clone()
  print (feat)

  torch.save(opt.path..'test.t7', {txt=feat, img=fea_txt})

  -- fea_txt[#fea_txt+1] = net_txt:forward(txt):clone()
end

