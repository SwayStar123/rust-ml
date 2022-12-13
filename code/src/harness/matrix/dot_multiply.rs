use crate::nn::matrix::Matrix;

#[tokio::test]
async fn matrix_dot_multiply() {
    let a = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Matrix::from(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let c = a.dot_multiply(&b);
    assert_eq!(c.data, vec![vec![5.0, 12.0], vec![21.0, 32.0]]);
}