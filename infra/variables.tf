variable "aws_region" {
  default = "eu-north-1"
}

variable "bucket_name" {
  description = "Globally unique S3 bucket name"
  type        = string
}