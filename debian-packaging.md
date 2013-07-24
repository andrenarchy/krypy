# Notes on Debian/Ubuntu packaging

Also check [krypy's packaging repo](https://github.com/andrenarchy/krypy-debian).

These notes rely on the [documentation of git-buildpackage](https://honk.sigxcpu.org/piki/projects/git-buildpackage/).

## First import
In upstream repo (replace HEAD by appropriate git tag, e.g. `v1.0.1`):
```
git archive --format=tar --prefix=krypy-0.1/ HEAD | gzip > /tmp/krypy_0.1.orig.tar.gz
mkdir ~/krypy-debian
cd ~/krypy-debian
git init
git-import-orig /tmp/krypy_0.1.orig.tar.gz
```
Enter
```
Package name: krypy
upstream version: 0.1
```
Then
```
 mkdir debian
 # fill debian/
 git add debian
 git commit
 gbp-create-remote-repo --remote-url-pattern='git@github.com:andrenarchy/krypy-debian.git'
```
Other maintainers can then do
```
gbp-clone git@github.com:andrenarchy/krypy-debian.git
```
Pulling can be done by
```
gbp-pull
```

## Upstream updates
In upstream repo:
```
git archive --format=tar --prefix=krypy-1.0.1 v1.0.1 | gzip > /tmp/krypy_1.0.1.orig.tar.gz
```
In debian repo:
```
git-import-orig /tmp/krypy_1.0.1.orig.tar.gz
# fiddle in debian/ (dch and friends)
git commit -a
git push
(git-buildpackage)
debuild -S
```
One level up:
```
dput ppa:andrenarchy/python krypy_1.0.0_source.changes
```
