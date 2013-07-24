# Notes on Debian/Ubuntu packaging

Also check [krypy's packaging repo](https://github.com/andrenarchy/krypy-debian).

These notes rely on the [documentation of git-buildpackage](https://honk.sigxcpu.org/piki/projects/git-buildpackage/).

## First import
```
git archive --format=tar --prefix=krypy-0.1/ HEAD | gzip > /tmp/krypy_0.1.orig.tar.gz
```
