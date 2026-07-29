terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

resource "aws_s3_bucket" "dvc_storage" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_public_access_block" "dvc_storage" {
  bucket                  = aws_s3_bucket.dvc_storage.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_iam_user" "dvc_mlops" {
  name = "dvc-mlops"
}

resource "aws_iam_user_policy" "dvc_mlops_s3" {
  name = "dvc-mlops-s3-access"
  user = aws_iam_user.dvc_mlops.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.dvc_storage.arn,
          "${aws_s3_bucket.dvc_storage.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_access_key" "dvc_mlops" {
  user = aws_iam_user.dvc_mlops.name
}