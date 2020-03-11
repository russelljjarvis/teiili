#!/bin/sh

# Called by "git push" after it has checked the remote status, but
# before anything has been pushed.  If this script exits with a non-zero status
# nothing will be pushed.
#
# This hook is called with the following parameters:
#
# $1 -- Name of the remote to which the push is being done
# $2 -- URL to which the push is being done
#
# If pushing without using a named remote those arguments will be equal.
#
# Information about the commits which are being pushed is supplied as lines to
# the standard input in the form:
#
#   <local ref> <local sha1> <remote ref> <remote sha1>
#
# This file prevent push of commits where the log message starts with "WIP"
# (work in progress) and in the case where the tests fails.

remote="$1"
url="$2"
z40=0000000000000000000000000000000000000000

# absolute path of the top-level directory
GIT_DIR=$(git rev-parse --show-toplevel)

while read local_ref local_sha remote_ref remote_sha
do
	if [ "$local_sha" = $z40 ]
	then
		# Handle delete
		:
	else
		if [ "$remote_sha" = $z40 ]
		then
			# New branch, examine all commits
			range="$local_sha"
		else
			# Update to existing branch, examine new commits
			range="$remote_sha..$local_sha"
		fi

		# Check for WIP commit
		commit=`git rev-list -n 1 --grep '^WIP' "$range"`
		if [ -n "$commit" ]
		then
			echo >&2 "Found WIP commit in $local_ref, not pushing"
			exit 1
		fi

		# Run the tests
		echo "Preparing to run unit tests... "
		for f in $GIT_DIR/tests/*.py;
			do python3 $f;
			result=$?;
		    	if [ $result -ne 0 ]
			then
				echo "Test failed. Please fix the errors. Nothing added to the push."
				exit 1
			fi
		done
	fi
done

exit 0
