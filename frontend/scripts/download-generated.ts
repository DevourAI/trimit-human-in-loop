// Import AWS SDK and promises
const AWS = require('aws-sdk');
const fs = require('fs');
const path = require('path');

// Configure the AWS SDK
AWS.config.update({
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  region: process.env.AWS_S3_BUCKET_REGION
});

const s3 = new AWS.S3();

function removePrefix(fullString: string, prefixToRemove: string): string {
  if (fullString.startsWith(prefixToRemove)) {
    return fullString.slice(prefixToRemove.length);
  }
  return fullString;
}

async function downloadFile(bucketName: string, prefix: string, key: string, localDestDir: string): Promise<void> {
  const params = {
    Bucket: bucketName,
    Key: key,
  };

  try {
    const data = await s3.getObject(params).promise();
    const localDest = path.join(localDestDir, removePrefix(key, prefix));
    fs.writeFileSync(localDest, data.Body as Buffer);
    console.log(`File downloaded successfully: ${key} -> ${localDest}`);
  } catch (err) {
    console.error('Failed to download file:', err);
    throw err;
  }
}

async function downloadAllFiles(bucketName: string, prefix: string, localDest: string): Promise<void> {
  const listParams = {
    Bucket: bucketName,
    Prefix: prefix
  };

  try {
    const listedObjects = await s3.listObjectsV2(listParams).promise();
    if (listedObjects === undefined || listedObjects.Contents === undefined) {
      throw new Error("could not list contents");
    }

    for (const obj of listedObjects.Contents) {
      if (obj !== undefined && obj.Key !== undefined) {
        await downloadFile(bucketName, prefix, obj.Key, localDest);
      }
    }

    if (listedObjects.IsTruncated) {
      await downloadAllFiles(bucketName, prefix, localDest);  // Recursive call to handle more than 1000 objects
    }
  } catch (err) {
    console.error('Failed to list files:', err);
    throw err;
  }
}

// Example usage: Downloading a specific file
// Modify to download different files as needed
async function main() {
  const bucketName = process.env.AWS_S3_BUCKET_NAME; // Set this to your S3 bucket name
  if (bucketName === undefined) {
    throw new Error("must set AWS_S3_BUCKET_NAME");
  }
  const prefix = 'frontend/';  // Set the folder path in S3
  const downloadPath = './gen'; // Local directory to save downloads

  await downloadAllFiles(bucketName, prefix, downloadPath);
}

main().catch(console.error);
