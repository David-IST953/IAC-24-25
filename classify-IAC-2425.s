# ===========================================================
# Identificacao do grupo:  A [T?? para Tagus ou A?? para Alameda]
#
# Membros [istID, primeiro + ultimo nome]
# 1. ist113618, Diogo Enes
# 2. ist114702, David Borges
# 3. ist114524, José Loureiro
#
# ===========================================================
# Requisitos do enunciado que *nao* estao corretamente implementados:
# (indicar um por linha, ou responder "nenhum")
# -
#
# ===========================================================
# Top-5 das otimizacoes que a vossa solucao incorpora:
# (maximo 140 caracteres por cada otimizacao)
#
# 1. Uso do slli em vez de mul (mais otimizado no risc-v)
#
# 2. Booleano no dotproduct que define se a função é chamada pelo matmul
#
# 3. Subtrair 32 no mesmo loop que convertemos m0 e m1 para 32 bits
#
# 4. Salto de 12 bytes simples na leitura do input (ignorar o cabeçalho do ficheiro)
#
# 5. Verificação do input dado
#
# ===========================================================

.data

# ===========================================================
#Main data structures. These definitions cannot be changed.

h_m0: .word 128
w_m0: .word 784
m0: .zero 401408                #h_m0 * w_m0 * 4 bytes

h_m1: .word 10
w_m1: .word 128
m1: .zero 5120                  #h_m1 * w_m1 * 4 bytes

h_input: .word 784
w_input: .word 1
input: .zero 3136               #h_input * w_input * 4 bytes

h_h: .word 128
w_h: .word 1
h: .zero 512                    #h_h * w_h * 4 bytes

h_o: .word 10
w_o: .word 1
o: .zero 40                     #h_o * w_o * 4 bytes


# ===========================================================
# Here you can define any additional data structures that your program might need

m0_file:     .string "classifier-files/weight-matrices/m0.bin"
m1_file:     .string "classifier-files/weight-matrices/m1.bin"
input_file:  .string "classifier-files/input-images/ascii-pgm/input0.pgm"

temp:      .zero 100352


# ===========================================================
.text

main:
    #m0
    la a0, m0_file
    la a1, m1_file
    la a2, input_file
    
    # Chamar classify
    jal ra, classify

    li a7, 1                 # PrintInt call
    ecall
    j exit
    

# ===========================================================
# FUNCTION: abs
#   Computes absolute value of the int stored at a0
# Arguments:
#   a0, a pointer to int
# Returns:
#   Nothing (modifies value in memory)
# ===========================================================
abs:
    lw t0, 0(a0)            # Load int value
    bge t0, zero, abs_done  # If value >= 0, skip negation (ERRO AQUI)
    sub t0, x0, t0          # t0 = -t0
    sw t0, 0(a0)            # Store back to memory
    
abs_done:
    jr ra                    # Return to the caller



# ============================================================
# FUNCTION: relu
#   Applies ReLU on each element of the array (in-place)
# Arguments:
#   a0 = pointer to int array
#   a1 = array length
# Exceptions:
#   - If the length of the array is less than 1,
#     this function terminates the program with error code 36
# ============================================================
relu:
    li t0, 0          # Iterador para cada endereço
    li t1, 0          # Salto do endereço
    li t5, 1          # Verificação do tamanho do array

    bge a1, t5, relu_loop # Se for menor que 1, sair com erro
    li a0, 36
    j exit_with_error

relu_loop:
    add t2, a0, t1            # Acede ao endereço de memória de cada valor
    lw t3, 0(t2)              # Copia o valor para t3
    bgt t3, x0, relu_loop_add # Caso for maior que 0, continuar para relu_loop_add
    sw x0, 0(t2)              # Se for menor, substituir o valor por 0

relu_loop_add:
    addi t1, t1, 4        # Adicionar +4 ao salto (salto para o próximo valor)
    addi t0, t0, 1        # Iterar por +1
    blt t0, a1, relu_loop # Loop enquanto t0 for menor que a1
    jr ra		  # normal return


# =================================================================
# FUNCTION: Given an int array, return the index of the largest
#   element. If there are multiple, return the one
#   with the smallest index.
# Arguments:
#   a0 (int*) is the pointer to the start of the array
#   a1 (int)  is the number of elements in the array
# Returns:
#   a0 (int)  is the first index of the largest element
# Exceptions:
#   - If the length of the array is less than 1,
#     this function terminates the program with error code 37
# =================================================================
argmax:
    li t0, 0               # Iterador para cada endereço
    li t1, 0               # Salto do endereço
    lw t4, 0(a0)           # Primeiro elemento do array
    li t5, 0               # Assume-se que o primeiro elemento é o maior
    li t6, 1               # Verificação do tamanho do array

    bge a1, t6, argmax_loop  # Se for menor que 1, sair com erro
    li a0, 37                # Exceção 37, o vetor tem tamanho menor que 1
    j exit_with_error

argmax_loop:
    add t2, a0, t1        	 # Acede ao endereço de memória de cada valor
    lw t3, 0(t2)          	 # Copia o valor para t3
    ble t3, t4, argmax_loop_add  # Se t3 for menor ou igual ao maior valor, segue-se a iteração
    mv t4, t3             	 # Caso for maior, registar o valor
    mv t5, t0             	 # E registar também o seu índice

argmax_loop_add:
    addi t1, t1, 4          # Adicionar +4 ao salto (salto para o próximo valor)
    addi t0, t0, 1          # Iterar por +1
    blt t0, a1, argmax_loop # Loop enquanto t0 for menor ou igual que a1
    mv a0, t5               # Substituição do a0 pelo índice do maior elemento
    jr ra                   # Return to the caller




# =======================================================
# FUNCTION: Dot product of 2 int arrays
# Arguments:
#   a0 (int*) - Pointer to the start of arr0
#   a1 (int*) - Pointer to the start of arr1
#   a2 (int)  - Number of elements to use
#   a7 (int)  - 1 caso for necessario fazer saltos maiores, 0 se for array normal
# Returns:
#   a0 (int)  - The dot product of arr0 and arr1
# Exceptions:
#   - If a2 < 1, exit with error code 38
# =======================================================
dotproduct:
    # Temporários
    li t0, 0                    # Iterador para cada endereço
    li t1, 0                    # Salto do endereço
    li t2, 0                    # Resultado do somatório
    li t6, 0                    # Primeiro valor do segundo array

    bgt a2, x0, dotproduct_loop # Se for maior que 0, seguir
    li a0, 38        		    # Exceção 38, o vetor tem tamanho menor que 1
    j exit_with_error
    

dotproduct_loop:
    add t3, a0, t1         # Acede ao endereço de memória do array1 de cada valor
    lw t3, 0(t3)           # Copia o valor do array1 para t4
    add t4, a1, t6         # Acede ao endereço de memória do array2 de cada valor
    lw t4, 0(t4)           # Copia o valor do array2 para t5
    mul t3, t3, t4         # Multiplica o valor de ambos os arrays
    add t2, t2, t3         # Adiciona ao somatório
    
    addi t1, t1, 4              # Shift do array1
    beq a7, x0, dotproduct_add1 # Shift normal do array2
    add t6, t6, t5              # Shift do array2 (caso do matmul)
    j dotproduct_itera

dotproduct_add1:
    addi t6, t6, 4              # Shift do array2

dotproduct_itera:
    addi t0, t0, 1              # Iterar por +1
    blt t0, a2, dotproduct_loop # Loop enquanto t0 for menor ou igual que a1
    mv a0, t2                   # Substituição do a0 pela soma de todas as multiplicações
    jr ra                       # Return to the caller


# =======================================================
# FUNCTION: Matrix Multiplication of 2 integer matrices
#   d = matmul(m0, m1)
#
# Arguments:
#   a0 (int*)  - pointer to the start of m0     (Matrix A)
#   a1 (int*)  - pointer to the start of m1     (Matrix B)
#   a2 (int)   - number of rows in m0 (A)             [rows_A]
#   a3 (int)   - number of columns in m0 (A)          [cols_A]
#   a4 (int)   - number of rows in m1 (B)             [rows_B]
#   a5 (int)   - number of columns in m1 (B)          [cols_B]
#   a6 (int*)  - pointer to the start of d            (Matrix C = A x B)
#
# Returns:
#   None (void); result is stored in memory pointed to by a6 (d)
#
# Exceptions:
#  - If the height or width of any of the matrices is less than 1, 
#    this function terminates the program with error core 39
#  - If the number of columns in matrix A is not equal to the number 
#    of rows in matrix B, it terminates with error code 40
# =======================================================
matmul:
  bne  a3, a4, erro_compativel # se COLS_A ≠ ROWS_B → vai direto ao erro
  ble  a2, x0, erro_dimens     # se ROWS_A ≤ 0    → erro
  ble  a3, x0, erro_dimens     # se COLS_A ≤ 0    → erro
  ble  a4, x0, erro_dimens     # se ROWS_B ≤ 0    → erro
  ble  a5, x0, erro_dimens     # se COLS_B ≤ 0    → erro
  
matmul_main:
    li    t0, 1      # Constante 1
    li    t1, 0      # Seletor de linha de A
    li    t2, 0      # Seletor de coluna de C

matmul_loop_linha:
    # Guardar na pilha
    addi  sp, sp, -32
    sw    a0, 0(sp)   # Endereço de A
    sw    a1, 4(sp)   # Endereço de B
    sw    a2, 8(sp)   # nº de linhas de A

    # Calcular offset de linha em bytes: t4 = a3 * 4
    slli  t4, a3, 2
    mv    t5, t1      # Cópia do índice da linha

matmul_argumentos:
    # Deslocar até à linha t1 de A
    beq   t5, x0, matmul_a1
    add   a0, a0, t4
    addi  t5, t5, -1
    j     matmul_argumentos

matmul_a1:
    # Preparar a1: deslocar até à coluna t2 de B
    li    a7, 1           # flag de dotproduct para saltos em arr2
    slli  t5, a5, 2       # salto entre colunas de B (a5 * 4)
    slli  t4, t2, 2       # deslocamento inicial (t2 * 4)
    add   a1, a1, t4

    # a2 = dimensão do dotproduct
    mv    a2, a3

    # Guardar temporários e chamar dotproduct
    sw    t0, 12(sp)
    sw    t1, 16(sp)
    sw    t2, 20(sp)
    sw    t3, 24(sp)

    # Chamar dotproduct
    sw    ra, 28(sp)
    jal   ra, dotproduct
    lw    ra, 28(sp)

    # Recarregar temporários
    lw    t3, 24(sp)
    lw    t2, 20(sp)
    lw    t1, 16(sp)
    lw    t0, 12(sp)
    mv    t6, a6
    mul   t4, t1, a5      # linhas * nº colunas de B
    add   t4, t4, t2      # + coluna
    slli  t4, t4, 2       # * 4 bytes por elemento
    add   t6, t6, t4
    sw    a0, 0(t6)

    # Restaurar argumentos iniciais
    lw    a2, 8(sp)
    lw    a1, 4(sp)
    lw    a0, 0(sp)
    addi  sp, sp, 32

    # Próxima coluna
    addi  t2, t2, 1
    beq   t2, a5, matmul_linha_feita
    j     matmul_loop_linha

matmul_linha_feita:
    # Próxima linha
    addi  t1, t1, 1
    li    t2, 0
    blt   t1, a2, matmul_loop_linha

    # Fim
    jr    ra

erro_compativel:
    li a0,40
    j exit_with_error
erro_dimens:
  li   a0, 38                 # ou outro código
  j    exit_with_error



######################################################################
# Function: read_file(char* filename, byte* buffer, int length)
# Input:
#   a0: pointer to null-terminated filename string
#   a1: destination buffer
#   a2: number of bytes to read
#   a3: 1 caso o ficheiro dado for pgm, 0 para leitura normal
# Output:
#   a0: number of bytes read (return value from syscall)
# Exceptions:
#   - Error code 41 if error in the file descriptor
#   - Error code 42 If the length of the bytes to read is less than 1
######################################################################

read_file:
    mv t1, a1                # Salvaguarda dos argumento a1
    li a1, 0                 # Flags do open
    li a7, 1024              # Ecall de open
    ecall
    bge a0, x0, read_ver_a2  # Se houver erro na abertura
    li a0, 41
    j exit_with_error

read_ver_a2:
    bgt a2, x0, read_leitura # Se o tamanho de bytes a ler for menor que 1
    li a0, 42
    j exit_with_error
    
read_leitura:
    mv a1, t1        	    # Buffer
    li a7, 63        	    # Ecall do read
    ecall
    bge a0, x0, read_fechar # Se houver erro na leitura
    li a0, 41
    j exit_with_error
    
read_fechar:
    li a7, 57       # Ecall do close
    ecall
    jr ra           # Return to the caller




# =======================================================
# FUNCTION: Classify decimal digit from input image
#   d = classify(A, B, input)
#
# Arguments:
#   a0 (int*)  - pointer to the start of weight matrix, m0
#   a1 (int*)  - pointer to the start of weight matrix, m1
#   a2 (int*)  - pointer to the start of input matrix
#
# Returns:
#   a0 (int) - value of the classified decimal digit
# Exceptions:
#   - Error code 43 if input is not type P5
#
# =======================================================

classify:
    addi sp,sp,-12
    sw ra, 0(sp)
    sw a0, 4(sp)
    sw a1, 8(sp)
    
    #input
    mv a0, a2
    la a1,temp
    li a2, 796
    jal ra, read_file
    
    la t0,temp
    lw t1, 1(t0)
    li t2, 53
    bne t1, t2, classify_image_not_P5 # Verificar caso o input dado não estiver no formato correto
    
    addi t0,t0,12
    la a0,input
    li a1,784
    li a2,0
    jal ra, classify_32bits_loop
    j classify_matmul_1
    
    
    #m0 começo
    lw a0, 4(sp)
    la a1, temp
    li a2, 100352
    jal ra, read_file
    
    la a0,m0
    li a1, 100352
    li a2,1
    jal ra, classify_32bits
    
    #m1 
    lw a0, 8(sp)
    la a1, temp
    li a2, 1280
    jal ra, read_file
    
    la a0,m1
    li a1,1280
    li a2,1
    jal ra, classify_32bits
    
classify_32bits:
    la t0,temp
    

classify_32bits_loop:
    lbu t1, 0(t0)
    beq a2,x0,classify_store
    addi t1,t1,-32
    
    
classify_store:
    sw t1, 0(a0)
    addi a0,a0,4
    addi t0,t0,1 
    addi a1,a1,-1 #iterador
    bgt a1,x0, classify_32bits_loop
    jr ra
    
classify_matmul_1:
    # Argumentos de matmul
    la a0, m0
    la a1, input
    li a2, 128
    li a3, 784
    li a4, 784
    li a5, 1
    la a6, h
    
    # Invocar matmul
    jal ra, matmul

    # Executar relu(h)

classify_relu:
    
    la a0, h
    li a1, 128
    
    # Invocar relu
    jal ra, relu

    # Computar o = matmul(m1, h)

classify_matmul_2:
    # Argumentos de matmul
    la a0, m1
    la a1, h
    li a2, 10
    li a3, 128
    li a4, 128
    li a5, 1
    la a6, o
    
    # Invocar matmul
    jal ra, matmul

    # Executar argmax(o)

classify_argmax:
    la a0, o
    li a1, 10

    # Invocar argmax
    jal ra, argmax
    
    # Retornamos o valor do ra na pilha
    lw ra, 0(sp)
    addi sp, sp, 12

    jr ra            # Return to the caller
    

classify_image_not_P5:
    li a0, 43
    j exit_with_error
    
# =======================================================
# Exit procedures
# =======================================================

# Exits the program (with code 0)
exit:
    li a7, 10     # Exit syscall code
    ecall         # Terminate the program

# Exits the program with an error 
# Arguments: 
# a0 (int) is the error code 
# You need to load a0 the error to a0 before to jump here
exit_with_error:
  li a7, 93            # Exit system call
  ecall                # Terminate program
