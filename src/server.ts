import { json as Json, urlencoded as Urlencoded } from "body-parser";
import * as cookieParser from "cookie-parser";
import * as express from "express";
import * as logger from "morgan";
import * as path from "path";
import * as errorHandler from "errorhandler";
import * as methodOverride from "method-override";
import { LandingRoute, UserRoute, DataPartialRoute, BerryAnalyzerRoute } from './routes';
import * as CORS from 'cors';
import * as Passport from 'passport';
import { AuthService, PassportConf } from './routes/auth';
import * as Session from 'express-session';
import * as ConnectMongo from 'connect-mongo';
//const MongoStore = ConnectMongo(Session);

const dust = require('express-dustjs');
const conf = require('../config/enviroment');

/**
 * The server.
 *
 * @class Server
 */
export class Server {

    public app: express.Application;

    /**
     * Bootstrap the application.
     *
     * @class Server
     * @method bootstrap
     * @static
     * @return {ng.auto.InjectorService} Returns the newly created injector for this app.
     */
    public static bootstrap(): Server {
        return new Server();
    }

    /**
     * Constructor.
     *
     * @class Server
     * @constructor
     */
    constructor() {

        //create expressjs application
        this.app = express();

        //configure application
        this.config();

        //add api
        this.api();

        //add routes
        this.routes();
    }

    /**
     * Create REST API routes
     *
     * @class Server
     * @method api
     */
    public api() {
        this.app.use('/api/partials', DataPartialRoute.route);
        this.app.use('/auth', AuthService.route);
        this.app.use('/api/users', AuthService.isAuthJWT('user'), UserRoute.route);
        this.app.use('/api/berries', BerryAnalyzerRoute.route);
    }

    /**
     * Configure application
     *
     * @class Server
     * @method config
     */
    public config() {
        //add headers
        this.app.use(function (req, res, next) {

            // Website you wish to allow to connect
            res.setHeader('Access-Control-Allow-Origin', '*');

            // Request methods you wish to allow
            res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');

            // Request headers you wish to allow
            res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type');

            // Set to true if you need the website to include cookies in the requests sent
            // to the API (e.g. in case you use sessions)
            //res.setHeader('Access-Control-Allow-Credentials', true);

            // Pass to next layer of middleware
            next();
        });
        //add static paths
        this.app.use(express.static(path.join(__dirname, "../public")));

        //configure dust
        this.app.engine("dust", dust.engine({ useHelpers: true }))
        this.app.set("views", path.join(__dirname, "../views"));
        this.app.set("view engine", "dust");

        //use logger middlware
        this.app.use(logger("dev"));

        //use json form parser middlware
        this.app.use(Json());

        //use query string parser middlware
        this.app.use(Urlencoded({ extended: true }));

        //use cookie parser middleware middlware
        this.app.use(cookieParser(conf.secrets.cookieParser));

        //use override middlware
        this.app.use(methodOverride());

        //configure session
        const optMC: ConnectMongo.MongoUrlOptions = { url: process.env.MONGODB_LOCAL };

        this.app.use(Session({
            secret: conf.secrets.session,
            resave: false,
            saveUninitialized: false,
            //store: new MongoStore(optMC)
        }));

        // PassportJs Configuration
        this.app.use(Passport.initialize());

        // setting JWT strategy
        PassportConf.JWTconf(Passport);

        // setting Local strategy
        PassportConf.LocalConf(Passport);
        this.app.use(Passport.session());


        //catch 404 and forward to error handler
        this.app.use(function (err: any, req: express.Request, res: express.Response, next: express.NextFunction) {
            err.status = 404;
            next(err);
        });

        //error handling
        this.app.use(errorHandler());

        //CORS
        this.app.use(CORS())

    }

    /**
     *
     * @class Server
     * @method routes
     */
    public routes() {

        this.app.use('/', LandingRoute.route);

        this.app.get("/secret", AuthService.isAuthJWT('editor'), function (req, res) {
            res.json({ message: "Success! You can not see this without a token", user: req.user, auth: req.isAuthenticated() });
        });

        this.app.get("/local-secret", AuthService.isAuth(), (req: express.Request, res: express.Response) => {
            res.json({ message: "Success! You can not see this without an auth", user: req.user, isAuth: req.isAuthenticated(), d: req.connection.localPort });
        });


    }

}