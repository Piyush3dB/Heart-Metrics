-- Import packages
require 'torch'

--------------------------------------
-- Define the HeartMetrics 'class'  --
--------------------------------------
-- Constructor
HeartMetrics = {}
HeartMetrics.__index = HeartMetrics
function HeartMetrics.create(name)
  
   -- initialize our object
   local instance = {}
   
   instance.name = name
   height = torch.Tensor({     0.11429,    0.15007,     0.73703,    0.07014,     0.1004,     0      ,   0.08135,     0.31554,    0      ,   0      ,    0      ,   0      ,   0      ,   0       })
   gender = torch.Tensor({    -1.45756,    0      ,    -9.47838,   -2.35620,    -2.62433,   -1.73000,   0      ,     0      ,   -2.03817,  -1.85918,   -8.55635,  -0.56512,  -0.53678,  -0.57818 })
   age    = torch.Tensor({    -0.04141,    0      ,    -0.33895,    0      ,    -0.03906,    0.09453,   0.02779,     0.10538,    0.08743,   0      ,    0      ,   0.02344,   0.02469,   0.01489 })
   weight = torch.Tensor({     0.07404,    0.07658,     0.42808,    0.04375,     0.09270,    0.11356,   0.05673,     0.21533,    0.10712,   0.07818,    0.27999,   0.02031,   0.01872,   0.01964 })

   -- Matrix M and Intercept Array
   instance.M = torch.cat({height, gender, age, weight}, 2)
   instance.I = torch.Tensor({{  20.98637,   15.38196,   -39.37731,   15.94580,    -0.97314,   34.23014,   -4.07280,  -34.73810,   35.52012,   9.15398,   18.97912,   7.66249,   7.34011,   7.49088 }}):t()
   
   -- make HeartMetrics handle lookup
   setmetatable(instance,HeartMetrics)
   return instance
end


-- 'compute' member function
function HeartMetrics:compute(x)
	-- Compute the metrics
   self.y = self.M*x + self.I

end


---------------------------------------------
-- HeartMetrics usage                      --
---------------------------------------------

-- Input data example
x = torch.Tensor({{ 166,  1,  71,  85 }}):t()
print("Input testcase ... ")
print(x)

-- create and use a HeartMetrics filter
HM = HeartMetrics.create("Simple HeartMetrics")

-- single iteration here
HM:compute(x)
print("Output metrics...")
print(HM.y)


