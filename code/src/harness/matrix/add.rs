use crate::nn::matrix::Matrix;

#[tokio::test]
async fn matrix_add() {
    let a = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Matrix::from(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let c = a.add(&b);
    assert_eq!(c.data, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
}

#[tokio::test]
async fn matrix_add_negative_nums() {
    let a = Matrix::from(vec![vec![-1.0, -2.0], vec![-3.0, -4.0]]);
    let b = Matrix::from(vec![vec![-5.0, -6.0], vec![-7.0, -8.0]]);
    let c = a.add(&b);
    assert_eq!(c.data, vec![vec![-6.0, -8.0], vec![-10.0, -12.0]]);
}