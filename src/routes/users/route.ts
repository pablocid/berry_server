import { BaseRoute } from '../../models/class.route'
import { NextFunction, Request, Response, Router } from "express";
import { TopBar } from "../../models/topbar";
import { UserController } from './controller';
import { AuthService } from '../auth';
import { extend as Extend } from 'lodash';
export class UserRoute extends BaseRoute {
    /**
     * Create the routes.
     *
     * @class UserRoute
     * @method route
     * @static
     */
    public static get route() {
        let r = Router();
        //get all users
        r.get("/", AuthService.isAuthJWT('admin'), (req: Request, res: Response, next: NextFunction) => {
            new UserRoute().index(req, res, next);
        });

        // get user by id
        r.get("/me", (req: Request, res: Response, next: NextFunction) => {
            new UserRoute().getMe(req, res, next);
        });

        // get user by id
        r.get("/:id", AuthService.isAuthJWT('admin'), (req: Request, res: Response, next: NextFunction) => {
            new UserRoute().getUser(req, res, next);
        });

        //create user
        r.post("/", AuthService.isAuthJWT('admin'), (req: Request, res: Response, next: NextFunction) => {
            new UserRoute().setUser(req, res, next);
        });

        //update user
        r.post("/me", AuthService.isAuthJWT('user'), (req: Request, res: Response, next: NextFunction) => {
            new UserRoute().updateMe(req, res, next);
        });

        //update users
        r.post("/:id", AuthService.isAuthJWT('admin'), (req: Request, res: Response, next: NextFunction) => {
            new UserRoute().update(req, res, next);
        });

        //delete user
        r.delete("/:id", AuthService.isAuthJWT('admin'), (req: Request, res: Response, next: NextFunction) => {
            new UserRoute().deleteUser(req, res, next);
        });

        return r;
    }

    /**
     * Constructor
     *
     * @class UserRoute
     * @constructor
     */
    private _userCtrl: UserController;

    constructor() {
        super();
        this._userCtrl = new UserController();
    }

    /**
     * The home page route.
     *
     * @class IndexRoute
     * @method index
     * @param req {Request} The express Request object.
     * @param res {Response} The express Response object.
     * @next {NextFunction} Execute the next method.
     */
    public async index(req: Request, res: Response, next: NextFunction) {
        try {
            var u1 = await this._userCtrl.users;
        } catch (e) {
            res.json(e);
            console.error(e.message);
        }

        res.json(u1.map(x => x.json()));
    }

    public async getMe(req: Request, res: Response, next: NextFunction) {
        let id = req.user._id;
        let u1;

        try {
            u1 = await this._userCtrl.getUser(id);
        } catch (e) {
            console.error(e.message);
        }

        if (!u1._id) {
            // no content
            res.status(204).json();
            return;
        }
        res.json(u1.json());

    }

    public async updateMe(req: Request, res: Response, next: NextFunction) {
        const _id = req.user._id;
        const email = req.body.email;
        const firstName = req.body.firstName;
        const lastName = req.body.lastName;

        let upData: any = { _id, email, firstName, lastName };
        for (let d in upData) {
            if (upData[d] === undefined) delete upData[d];
        }
       
        let Up;
        try {
            Up = await this._userCtrl.updateUser(upData);
        } catch (e) {
            console.log('catching error: ',e)
            res.status(400).json(e);
        }

        res.json(Up.json());

    }

    public async update(req: Request, res: Response, next: NextFunction) {

        const _id = req.params.id;
        const email = req.body.email;
        const firstName = req.body.firstName;
        const lastName = req.body.lastName;
        const password = req.body.password;
        const role = req.body.role;

        let upData: any = { _id, email, firstName, lastName, password, role };
        for (let d in upData) {
            if (upData[d] === undefined || upData[d] === '') delete upData[d];
        }
       
        let Up;
        try {
            Up = await this._userCtrl.updateUser(upData);
        } catch (e) {
            console.log('catching error: ',e)
            res.status(400).json(e);
        }

        res.json(Up.json());

    }

    public getUser(req: Request, res: Response, next: NextFunction) {
        let id = req.params.id;

        (async () => {
            try {
                var u1 = await this._userCtrl.getUser(id);
                if (!u1._id) {
                    // no content
                    res.status(204).json();
                    return;
                }
                res.json(u1.json());
            } catch (e) {
                console.error(e.message);
                res.status(400).json({ message: e.message });
            }
        })();

    }

    public async setUser(req: Request, res: Response, next: NextFunction) {
        const email = req.body.email;
        const firstName = req.body.firstName;
        const lastName = req.body.lastName;
        const role = req.body.role;
        const password = req.body.password

        let user = { email, firstName, lastName, role, password };
        let old

        try {
            old = await this._userCtrl.getUserByEmail(email);
        } catch (e) {
            console.error(e.message);
            res.status(400).json({ message: e.message });
        }

        if (old._id) {
            res.status(409).json({ message: 'the email already exist' });
            return;
        }

        try {
            res.status(201).json(await this._userCtrl.setUser(user));
        } catch (e) {
            console.error(e.message);
            res.status(400).json({ message: e.message });
        }
    }

    public async deleteUser(req: Request, res: Response, next: NextFunction) {
        let self = this;
        const id = req.params.id;
        let d;

        try {
            d = await self._userCtrl.deleteUser(id);
        } catch (e) {
            res.status(400).json({ message: e.message });
            return;
        }

        if (d._id) {
            res.json(d.json());
            return;
        }
        res.status(400).json(d);
    }
}