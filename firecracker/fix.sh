#!/bin/bash
set -e
 
# Clean out /var/cache/apt/archives
apt-get clean
# Fill it with all the .debs we need
apt-get --reinstall -dy install $(dpkg --get-selections | grep '[[:space:]]install' | cut -f1)
 
DIR=$(mktemp -d -t info-XXXXXX)
for deb in /var/cache/apt/archives/*.deb
do
    # Move to working directory
    cd "$DIR"
    # Create DEBIAN directory
    mkdir -p DEBIAN
    # Extract control files
    dpkg-deb -e "$deb"
    # Extract file list, fixing up the leading ./ and turning / into /.
    dpkg-deb -c "$deb" | awk '{print $NF}' | cut -c2- | sed -e 's/^\/$/\/./' > DEBIAN/list
    # Figure out binary package name
    DEB=$(basename "$deb" | cut -d_ -f1)
    # Copy each control file into place
    cd DEBIAN
    for file in *
    do
        cp -a "$file" /var/lib/dpkg/info/"$DEB"."$file"
    done
    # Clean up
    cd ..
    rm -rf DEBIAN
done
rmdir "$DIR"


