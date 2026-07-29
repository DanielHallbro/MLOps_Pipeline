output "dvc_access_key_id" {
  value = aws_iam_access_key.dvc_mlops.id
}

output "dvc_secret_access_key" {
  value     = aws_iam_access_key.dvc_mlops.secret
  sensitive = true
}