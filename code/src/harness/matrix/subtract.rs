use crate::nn::matrix::Matrix;

#[tokio::test]
async fn matrix_subtract() {
    let a = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Matrix::from(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let c = a.subtract(&b);
    assert_eq!(c.data, vec![vec![-4.0, -4.0], vec![-4.0, -4.0]]);
}