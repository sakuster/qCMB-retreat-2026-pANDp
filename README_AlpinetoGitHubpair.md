#Prion & Prejudice

## Connecting GitHub and Alpine
_These instructions were originally compiled by Eric Anderson._


- Logging into Alpine

Go to https://ondemand-rmacc.rc.colorado.edu/ or:
```sh
ssh CSUeid@colostate.edu@login11.rc.colorado.edu

# password is eidpassword,push
# (you have to add ,push to the end, then use the DUO app)
```
- Setting up `git` on Alpine:
```sh
git config --global user.name "Your Name"
git config --global user.email "your.email@your.email.com"
git config --global core.editor nano
```
- setting up SSH key pair on Alpine for GitHub
```
# if you already have ~/.ssh/id_ed25519 and  ~/.ssh/id_ed25519.pub
# then you don't have to set these up, just go to the next step.
# If not, then it is simple, do this:

ssh-keygen -t ed25519 -C "FOR GITHUB"

# when prompted, save in default location and leave password
# blank by just hitting return.
```
- Copy the public key and put it on GitHub
```sh
cat ~/.ssh/id_ed25519.pub
# then copy it and go to GitHub->Settings->SSH and GPG keys
# and add the Key.
```
- Add a command to your `~/.bashrc` to wake up the ssh daemon
on the CURC.  Do `nano ~/.bashrc` and add the following lines to
the file and then save it:
```sh
alias gitup='eval "$(ssh-agent -s)"; ssh-add ~/.ssh/id_ed25519'
```
- Source your `~/.bashrc` and then run the gitup command.
```sh
source ~/.bashrc
gitup
```
- Test your connection to GitHub:
```sh
ssh -T git@github.com
```
