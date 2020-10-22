/**
 * @author Karan Gandhi
 * @email karangandhi.programming@gmail.com
 * @desc [description]
 */

class Matrix {
    // Constructor to accept the no of rows and cols for creating the matrix object
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.matrix = [];

        // initialise all the values of the matrix to 0
        for (var i = 0; i < rows; i++) {
            this.matrix[i] = [];
            for (var x = 0; x < cols; x++) {
                this.matrix[i][x] = 0;
            }
        }
    }

    // Give random values to the matrix
    randomSize() {
        for (var i = 0; i < this.rows; i++) {
            for (var x = 0; x < this.cols; x++) {
                this.matrix[i][x] = Math.random() * 2 - 1;
            }
        }
    }

    // Convert an array to a Matrix
    static fromArray(arr) {
        var arrMatrix = new Matrix(arr.length, 1);

        // Setting all the data in the array in a sepreate row of a new Matrix
        for (var i = 0; i < arr.length; i++) {
            arrMatrix.matrix[i][0] = arr[i];
        }
        return arrMatrix;
    }

    // Converts the matrix to an array
    toArray() {
        var matrixArray = [];

        for (var i = 0; i < this.rows; i++) {
            for (var x = 0; x < this.cols; x++) {
                matrixArray.push(this.matrix[i][x]);
            }
        }
        return matrixArray;
    }

    // Static multiply function which would multuply a matrix with another
    static multiply(m1, m2) {
        if (m1 instanceof Matrix && m2 instanceof Matrix) {
            // Matrix Product
            if (m1.cols !== m2.rows) {
                // Check if cols of 'a' is not equal to rows of 'b'
                console.log("Columns of A must match columns of B");
                // Just get out of here
                return undefined;
            }

            var result = new Matrix(m1.rows, m2.cols);
            var a = m1;
            var b = m2;

            for (var i = 0; i < result.rows; i++) {
                for (var x = 0; x < result.cols; x++) {
                    // Dot product of values in col
                    var sum = 0;
                    var val = 0;
                    for (var k = 0; k < a.cols; k++) {
                        val = a.matrix[i][k] * b.matrix[k][x];
                        sum += val;
                    }
                    result.matrix[i][x] = sum;
                }
            }
            return result;
        } else {
            return undefined;
        }
    }

    // Multiply
    multiply(n) {
        if (n instanceof Matrix) {
            // Hadamard product
            for (var i = 0; i < this.rows; i++) {
                for (var x = 0; x < this.cols; x++) {
                    this.matrix[i][x] *= n.matrix[i][x];
                }
            }
        } else {
            // Scalar multiplication
            for (var i = 0; i < this.rows; i++) {
                for (var x = 0; x < this.cols; x++) {
                    this.matrix[i][x] *= n;
                }
            }
        }
    }

    // Add
    add(n) {
        if (n instanceof Matrix) {
            // Matrix sum
            for (var i = 0; i < this.rows; i++) {
                for (var x = 0; x < this.cols; x++) {
                    this.matrix[i][x] += n.matrix[i][x];
                }
            }
        } else {
            // Scalar sum
            for (var i = 0; i < this.rows; i++) {
                for (var x = 0; x < this.cols; x++) {
                    this.matrix[i][x] += n;
                }
            }
        }
    }

    // Subtract
    static subtract(a, b) {
        // Return a new Matrix a - b
        var result = new Matrix(a.rows, a.cols);

        for (var i = 0; i < a.rows; i++) {
            for (var x = 0; x < a.cols; x++) {
                result.matrix[i][x] = a.matrix[i][x] - b.matrix[i][x];
            }
        }
        return result;
    }

    static transpose(matrix) {
        // Transposing the rows of the Matrix to the cols of the matrix and the cols of the Matrix to the rows of the Mattrix
        var result = new Matrix(matrix.cols, matrix.rows);
        for (var i = 0; i < matrix.rows; i++) {
            for (var x = 0; x < matrix.cols; x++) {
                result.matrix[x][i] = matrix.matrix[i][x];
            }
        }
        return result;
    }

    // Map function which performs any operation on the Matrix
    map(func) {
        // Apply a function to every element of the matrix
        for (var i = 0; i < this.rows; i++) {
            for (var x = 0; x < this.cols; x++) {
                var val = this.matrix[i][x];
                this.matrix[i][x] = func(val, i, x);
            }
        }
    }

    // static version of the map function
    static map(matrix, func) {
        var result = new Matrix(matrix.rows, matrix.cols);

        // Apply a function to every element of the matrix
        for (var i = 0; i < this.rows; i++) {
            for (var x = 0; x < this.cols; x++) {
                var val = this.matrix[i][x];
                this.matrix[i][x] = func(val);
            }
        }
        return matrix;
    }

    // Print the current Matrix
    printMatrix() {
        console.table(this.matrix);
    }
}
