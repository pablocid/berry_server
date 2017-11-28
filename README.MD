# ExpressCeed

## Cableado

- task runner: npm
- compiler: Typscript
- watcher: when .ts or view is modify > compile tsc (watching) and move files to dist folder > then nodemon reload the server > then browserSync reload the page.
  - server-side: nodemon with config file
  - client-side: browserSync with config file
  - compiler: tsc --watch with config file

## Enviromental variables

- DOMAIN
- SESSION_SECRET
- SECRET_SESSION
- COOKIEPARSER
- FACEBOOK_ID
- FACEBOOK_SECRET
- TWITTER_ID
- TWITTER_SECRET
- GOOGLE_ID
- GOOGLE_SECRET
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- DEBUG
- MONGODB_LOCAL
- APP_SECRET: utilizado para firmar jwt