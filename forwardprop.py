import numpy as np

def forwardPass(inputs, weight, bias):
    w_sum = np.dot(inputs, weight) + bias
    
    #Linear Actiovation f(x) = x
    act = w_sum
    
    return act

#pre-Trained Weight & Biases after Training
W= np.array([[2.9999999999]])
b= np.array([[1.9999999999]])

#initilize Input Data
inputs = np.array([[7],[8],[9],[10]])

# Output of Output Layer
o_out = forwardPass(inputs,W,b)

print('Output Layer Output (Linier)')
print('============================')
print(o_out, "\n")
Output Layer Output (Linier)
============================
[[23.]
 [26.]
 [29.]
 [32.]] 