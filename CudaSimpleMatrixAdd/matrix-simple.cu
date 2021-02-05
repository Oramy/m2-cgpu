#include <fstream>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

// Charge une matrice disponible dans les repertoires exemples
bool load_matrix(char * filename, float * &matrix, int &nx, int &ny){
  std::string line;
  std::ifstream infile(filename);

  if (!infile.is_open()) {
    std::cout << "Fichier introuvable: "<< filename << std::endl;
    return 0;
  }

  // Charge la taile de la matrice
  infile >> nx >> ny;

  // Alloue le tableau correspondant
  matrix = new float[nx*ny];

  // Charge la matrice
  for (int i=0; i< nx*ny; i++){
    infile >> matrix[i];
  }

  infile.close();

  return 1;
}

// Calcul C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
             int numARows, int numAColumns,
             int numBRows, int numBColumns,
             int numCRows, int numCColumns) {
    /// Insérer le code
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < numCColumns && y < numCRows) {
        int i = y * numCColumns + x;
        float s = 0;
        for (int k = 0; k < numAColumns; k++) {
            s += A[y * numAColumns + k] * B[k * numBColumns + x];
        }
        C[i] = s;
    }
}

int main(int argc, char** argv) {
    float* hostA;
    float* hostB;
    float* hostC;
    float* hostExpectedOutput;
    float* deviceA;
    float* deviceB;
    float* deviceC;
    int numARows;
    int numAColumns;
    int numBRows;
    int numBColumns;
    int numCRows;
    int numCColumns;
    int numORows;
    int numOColumns;



    /// Charger le fichier d'entree
    load_matrix(argv[1], hostA, numARows, numAColumns);
    load_matrix(argv[2], hostB, numBRows, numBColumns);
    if (numAColumns != numBRows){
        std::cerr << "Loaded matrix are not compatible: their dimensions are: "
        << "(" << numARows << ", " << numAColumns << ") and (" << numBRows << ", " << numBColumns;
    }
    /// Initialiser numCRows et numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    /// Allouer hostC
    hostC = new float[numCRows * numCColumns];
    /// Afficher les informations sur la matrice
    std::cout << "(" << numARows << ", " << numAColumns << ") x (" << numBRows << ", "
            << numBColumns << ") = ("
            << numCRows << ", " << numCColumns << ")" << std::endl;
    /// Allouer la memoire sur GPU
    cudaMalloc((void**)&deviceA, sizeof(float) * numARows * numAColumns);
    cudaMalloc((void**)&deviceB, sizeof(float) * numBRows * numBColumns);
    cudaMalloc((void **)&deviceC, sizeof(float) * numCRows * numCColumns);

    /// Copier la memoire sur le GPU
    cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice);


    /// Initialise la grille et les dimensions de chaque bloc
    int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 dim((int)(ceil((float)(numCRows) / block_size)),
             (int)(ceil((float)(numCColumns) / block_size)));

    std::cout << "Block size: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
    std::cout << "Grid size: (" << dim.x << ", " << dim.y << ", " << dim.z << ")" << std::endl;
    /// Execute le kernel
    matrixMultiply<<<dim, block>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns,
        numCRows, numCColumns);

    cudaDeviceSynchronize();


    /// Charge le resultat en memoire CPU
    cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns,  cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    load_matrix(argv[3], hostExpectedOutput, numORows, numOColumns);

    if (numOColumns != numCColumns
        || numORows != numORows) {
        std::cerr << "Output matrix have wrong dimensions" << std::endl;
        std::cerr << "(" << numORows << ", " << numOColumns << ") != ("
            << numCRows << ", " << numCColumns << ")" << std::endl;
    }

    float error = 0;
    for (int i = 0; i < numCColumns * numCRows; i++) {
        error += (hostExpectedOutput[i] - hostC[i]) * (hostExpectedOutput[i] - hostC[i]);
    }
    error /= (float)(numCColumns * numCRows);

    
    /// Libere la memoire
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    delete hostExpectedOutput;
    delete hostA;
    delete hostB;
    delete hostC;

    return 0;
}

