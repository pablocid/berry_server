import { Document, Model } from "mongoose";

export interface IUser {
  email?: string;
  firstName?: string;
  lastName?: string;
  password?: string;
}

export interface IUserModel extends IUser, Document {
  //custom methods for your model would be defined here
}

export interface IModel {
  user: Model<IUserModel>;
}