use crate::nn::matrix::Matrix;

#[tokio::test]
async fn matrix_transpose() {
    let mut a = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = a.transpose();
    assert_eq!(b.data, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
}