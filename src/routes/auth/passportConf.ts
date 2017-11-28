const conf = require('../../../config/enviroment');
import { ExtractJwt, Strategy as JWTStrategy, StrategyOptions } from 'passport-jwt';
import { Strategy as LocalStrategy } from 'passport-local';
import { sign as Sign, SignOptions } from 'jsonwebtoken';
import { UserController } from '../users';

export class PassportConf {

    public static JWTconf(P: any): void {
        let pass = new PassportConf();
        P.use(pass.jwtConfiguration());
    }

    public static SignJWT(payload: any): string {
        let signOpts: SignOptions = {};
        signOpts.expiresIn = '12h';
        return Sign(payload, conf.secrets.app, signOpts);
    }

    public static LocalConf(P: any): void {
        let pass = new PassportConf();
        P.use(pass.localConfiguration());
        P.serializeUser(pass.serializeUser());
        P.deserializeUser(pass.deserializeUser());
    }

    private userCtrl: UserController;
    
    constructor() {
        this.userCtrl = new UserController();
    }

    public jwtConfiguration() {
        let jwtOpts: StrategyOptions = { jwtFromRequest: ExtractJwt.fromAuthHeader(), secretOrKey: conf.secrets.app };

        let jwtConf = new JWTStrategy(jwtOpts, (jwt_payload, next) => {

            if (jwt_payload) next(null, jwt_payload);
            else next(null, false);
        });

        return jwtConf;
    }

    public localConfiguration() {
        let local = new LocalStrategy({
            usernameField: 'username',
            passwordField: 'password'
        }, (username, password, done) => {

            this.userCtrl.getUserByEmail(username)
                .then(u => {
                    if (this.isValidPassword(password, u.password))
                        done(null, u.json());
                    else
                        done(null, false);
                })
                .catch(e => done(e));
        });

        return local;
    }

    // Serialize user
    public serializeUser() {
        return (user: any, done: any) => {
            //console.log('serializeUser', user._id);
            done(null, user._id)
        }
    }

    //Deserialize user
    public deserializeUser() {
        return (id: string, done: any) => {
            this.userCtrl.getUser(id)
                .then(u => done(null, u.json()))
                .catch(e => done(e));
        }
    }

    public isValidPassword(candidate: string, password: string): boolean {
        //return bCrypt.compareSync(password, user.password);
        return true;
    }
}



