class NeuralNetwork {
    constructor (inputs, hidden, outputs, learning_rate) {
        // hidden, input and output nodes
        this.inputs_nodes = inputs;
        this.hidden_nodes = hidden;
        this.outputs_nodes = outputs;

        // Weights for data between inputs and hidden layers
        this.weights_ih = new Matrix(this.hidden_nodes, this.inputs_nodes);
        // Weights for data between hidden and output layers
        this.weights_ho = new Matrix(this.outputs_nodes, this.hidden_nodes);

        // Giving random Weights
        this.weights_ih.randomSize();
        this.weights_ho.randomSize();

        // Bias for the hidden layer 
        this.bias_h = new Matrix(this.hidden_nodes, 1);
        // Bias for the output layer 
        this.bias_o = new Matrix(this.outputs_nodes, 1);

        // Randomise the bias of the hidden and the output layer
        this.bias_h.randomSize();
        this.bias_o.randomSize();

        // setting the learning rate
        this.learning_rate = learning_rate; 
    }
		
    feedFoward(input_arr) {
        var input = Matrix.fromArray(input_arr);

        // Generating hidden outputs
        var hidden = Matrix.multiply(this.weights_ih, input);
        hidden.add(this.bias_h);
        // Activation function
        hidden.map(sigmoid);

        // Generating outputs for the output layer
        var output = Matrix.multiply(this.weights_ho, hidden);
        output.add(this.bias_o);
        // Activation function
        output.map(sigmoid);

        // Returning an output which is converted to an array
        return output.toArray();
    }

    // Train the Nerual network to give the desirable output
    train(input_arr, targets_arr) {
        // var output = this.feedFoward(inputs);

        var input = Matrix.fromArray(input_arr);

        // Generating hidden outputs
        var hidden = Matrix.multiply(this.weights_ih, input);
        hidden.add(this.bias_h);
        // Activation function
        hidden.map(sigmoid);

        // Generating outputs for the output layer
        var output = Matrix.multiply(this.weights_ho, hidden);
        output.add(this.bias_o);
        // Activation function
        output.map(sigmoid);

        // Converting array to Matrix object
        var targets = Matrix.fromArray(targets_arr);

        // Calculating the error
        // ERR = TARGETS - OUTPUTS
        var output_err = Matrix.subtract(targets, output);

        // CHANGE IN SLOPE FOR Y WHERE Y = MX + B
        // ▲ M = LEARNING_RATE * ERROR * X
        // ▲ B = LEARNING_RATE * ERROR

        // Calculating the gradient
        var gradiant = Matrix.map(output, dsigmoid);
				
        gradiant.multiply(output_err);
        gradiant.multiply(this.learning_rate);

        // Calculate deltas
        var hidden_t = Matrix.transpose(hidden);
        var weights_ho_delta = Matrix.multiply(gradiant, hidden_t);

        // Adjust the weights by the deltas
        this.weights_ho.add(weights_ho_delta);
        // Adjust the bias by its weights (which is the gradient)
        this.bias_o.add(gradiant);

        // Calculate the hidden layer outputs
        var weight_ho_transposed = Matrix.transpose(this.weights_ho);
        var hidden_errs = Matrix.multiply(weight_ho_transposed, output_err);

        // Calculate hidden gradient
        var hidden_gradient = Matrix.map(hidden, dsigmoid);
        hidden_gradient.multiply(hidden_errs);
        hidden_gradient.multiply(this.learning_rate);

        // Calculate input -> hidden deltas
        var input_t = Matrix.transpose(input);
        var weights_ih_delta = Matrix.multiply(hidden_gradient, input_t);

        // Adjust the weights by the deltas        
        this.weights_ih.add(weights_ih_delta);
        // Adjust the bias by its weights (which is the gradient)
        this.bias_h.add(hidden_gradient);

        // hidden_errs.printMatrix();
        // output.printMatrix();
        // targets.printMatrix();
        // err.printMatrix();
    }
}

// Sigmoid function
// Takes any value and converts it to a number between 0 and 1
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// Drivid of sigmoid
function dsigmoid(y) {
    return y * (1 - y);
}