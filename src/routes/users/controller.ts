import { Model } from "mongoose";
import { IUser, IUserModel } from "./interface";
//import { MongoConnection } from '../../dataconnection/mongodb';
import { UserSchema } from './schema';
import { User } from './class';
import { clone } from 'lodash';
import { AuthService } from '../auth';
import { defer as Defer } from 'q';
import { extend as Extend } from 'lodash';

export class UserController {
  public userModel: Model<IUserModel>;

  constructor() {
    //this.userModel = new MongoConnection('User', UserSchema).model;
  }

  public get users(): Promise<User[]> {
    return this.userModel.find()
      .then(u => {
        return u.map(x => new User(x));
      })
      .catch(e => { throw e; });

  }

  public getUser(id: string): Promise<User> {
    return this.userModel.findOne({ _id: id })
      .then(x => {
        let usr = new User(x);
        return usr;
      })
      .catch(e => { throw e; });
  }

  public setUser(user: any): Promise<User> {

    return AuthService.hashPassword(user.password)
      .then(hashPass => {
        if (!user.password) throw { message: 'no password' };
        user.password = hashPass;
        return user;
      })
      .then(u => {
        return this.userModel.create(u);
      })
      .then(x => {
        let usr = new User(x);
        return usr;
      })
      .catch(e => { throw e; });
  }

  public updateUser(value: any): Promise<User> {
    if (!value._id) throw { message: '_id dont exist' };

    return this.userModel.findById(value._id)
      .then(usr => {
        if (!usr || !usr._id) throw { message: 'user dont exist' };
        return usr;
      })
      .then(usr => {
        if (value.password) {
          return AuthService.hashPassword(value.password)
            .then(hashPass => {
              value.password = hashPass;
              return usr;
            })
            .catch(e => { throw e; });
        }
        return usr;
      })
      .then(usr => {
        if (value.email && value.email !== usr.email) {
          console.log('changind the email', usr.email, value.email);
          return this.userModel.findOne({ email: value.email })
            .then(exist => {
              if (exist) throw { message: `the email ${value.email} is already in use, pick another` };
              return usr;
            })
            .catch(e => { throw e; })
        }

        return usr;
      })
      .then(usr => Extend(usr, value))
      .then(usr => {
        return usr.save()
          .catch( (e:any) => { throw e; });
      })
      .then(usr => new User(usr))
      .catch(e => { throw e; });

  }

  public deleteUser(id: string): Promise<User> {
    return this.userModel.findByIdAndRemove(id)
      .then((x) => {
        let usr = new User(x);
        if (!usr._id) {
          throw { message: '_id is not found' };
        }
        return usr;
      })
      .catch(e => { throw e; });
  }

  public getUserByFirstName(value: string): Promise<User> {
    return this.userModel.findOne({ firstName: value })
      .then(x => {
        let usr = new User(x);
        return usr;
      })
      .catch(e => { throw e; });
  }

  public getUserByEmail(value: string): Promise<User> {
    return this.userModel.findOne({ email: value })
      .then(x => {
        let usr = new User(x);
        return usr;
      })
      .catch(e => { throw e; });
  }
}