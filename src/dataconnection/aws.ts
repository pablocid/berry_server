import { S3 } from 'aws-sdk';
import * as Q from 'q';

var conf = require('../../config/enviroment');

export class AWSConnection {
    public s3: S3;
    public bucket: string;
    constructor() {
        this.s3 = new S3();
        this.s3.config.accessKeyId = conf.AWS.accessKeyId;
        this.s3.config.secretAccessKey = conf.AWS.AWS_SECRET_ACCESS_KEY;
        this.s3.config.s3BucketEndpoint;
        this.s3.config.setPromisesDependency(Q);

        this.bucket = 'myexpress';
    }

    public get listBuckets() {
        return this.s3.listBuckets().promise();
    }

    public getBucketFiles() {
        var d = Q.defer();
        this.s3.listObjects({Bucket:this.bucket},(err, data)=>{
            if(err){ d.reject(err)}
            d.resolve(data);
        });
        return d.promise;
    }

}