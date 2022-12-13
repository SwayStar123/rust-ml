use crate::nn::matrix::Matrix;

#[tokio::test]
async fn matrix_multiply() {
    let a = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Matrix::from(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let c = a.multiply(&b);
    assert_eq!(c.data, vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
}