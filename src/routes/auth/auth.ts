import { BaseRoute } from '../../models/class.route'
import { NextFunction, Request, Response, Router } from 'express';
import * as Passport from 'passport';
import { PassportConf } from './passportConf';
import { Handler } from 'express';
import { UserController } from '../users';
import { hash as Hash, hashSync as HashSync, compareSync as CompareSync, compare as Compare } from 'bcrypt';
const _ = require('lodash');
const conf = require('../../../config/enviroment');

export class AuthService extends BaseRoute {
    /**
     * Create the routes.
     *
     * @class AuthService
     * @method route
     * @static
     */
    public static get route() {

        let r = Router();
        var obj = new AuthService();

        r.get("/", (req: Request, res: Response, next: NextFunction) => {
            obj.index(req, res, next);
        });

        r.post('/login', (req: Request, res: Response, next: NextFunction) => {
            obj.login(req, res, next);
        });

        r.post('/session-login', Passport.authenticate('local'), (req: Request, res: Response) => {
            res.json({ message: "ok", req: req.user, isAut: req.isAuthenticated() });
        });

        r.get("/logout", (req: Request, res: Response, next: NextFunction) => {
            let bef = req.isAuthenticated();
            req.logout();
            res.json({ message: 'Succesfully logout', isAuth: req.isAuthenticated(), isBefore: bef })
        });

        return r;
    }

    public static isAuth(role?: string): Handler {
        if (!role) role = 'guest';

        return (req: Request, res: Response, next: NextFunction) => {
            if (req.isAuthenticated() && AuthService.isRole(role, req.user.role))
                return next();

            res.status(401).json({ message: 'no auth' });

        }
    }

    public static isAuthJWT(role?: string): Handler {
        if (!role) role = 'guest';
        //console.log('isAuhJWT');
        let opt: Passport.AuthenticateOptions = {};
        opt.session = false;

        return (req: Request, res: Response, next: NextFunction) => {
            Passport.authenticate('jwt', opt, (nose: any, user: any, info: any, status: any) => {
                //console.log(nose, user, info, status);
                if (user && AuthService.isRole(role, user.role)) {
                    req.user = user;
                    return next();
                }

                res.status(401).json({ message: info ? info.message : 'no auth' });
            })(req, res, next);
        }

    }

    public static roles: string[] = ['guest', 'user', 'editor', 'admin'];

    private static isRole(role: string, userRole: string): boolean {
        let ri = AuthService.roles.indexOf(role);
        let uri = AuthService.roles.indexOf(userRole);

        if (!role || ri === -1) return false;
        if (!userRole || uri === -1) return false;

        if (uri >= ri) return true;
        else return false;
    }

    public static comparePasswords(candidate: string, hash: string): Promise<boolean> {
        return Compare(candidate, hash);
    }

    public static hashPassword(password: string): Promise<string> {
        return Hash(password, 10);
    }

    public users: any[];
    public userCtrl: UserController;

    /**
     * Constructor
     *
     * @class AuthService
     * @constructor
     */

    constructor() {
        super();
        this.userCtrl = new UserController();
    }

    /**
     *
     * @class AuthService
     * @method index
     * @param req {Request} The express Request object.
     * @param res {Response} The express Response object.
     * @next {NextFunction} Execute the next method.
     */
    public index(req: Request, res: Response, next: NextFunction): void {
        //lista de los routes disponibles
        res.json({ lista: [1, 2, 3, 4, 5] });
    }

    /**
     *
     * @class AuthService
     * @method login
     * @description login for jwt
     * @param req {Request} The express Request object.
     * @param res {Response} The express Response object.
     * @next {NextFunction} Execute the next method.
     */
    public login(req: Request, res: Response, next: NextFunction): void {
        const name = req.body.username;
        const password = req.body.password;
        if (!req.body.username || !req.body.password) {
            res.status(401).json({ message: 'no name or password set', isAuth: req.isAuthenticated() });
            return;
        }

        this.userCtrl.getUserByEmail(name)
            .then(u => {
                if (!u._id) throw "username did not match";
                return AuthService.comparePasswords(password, u.password)
                    .then(match => {
                        return { u, match };
                    })
            })
            .then(r => {
                if (!r.match) throw "passwords did not match";
                var token = PassportConf.SignJWT(r.u.payload());
                res.json({ message: "ok", token: token, isAuth: req.isAuthenticated() });

            }).catch(e => {
                res.status(400).json({ message: e });
            });
    }


}